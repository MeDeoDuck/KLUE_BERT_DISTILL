"""Fine-tune KLUE-RoBERTa-base on labeled comments.

Tuned for NVIDIA A40 (48GB, Ampere): bf16 autocast, TF32 matmul, batch 64.

Run:  python -m local_classifier.train
Env:  TRANSFORMERS_NO_ADVISORY_WARNINGS=1 helps quiet output.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from local_classifier import config as C
from local_classifier.dataset import CommentDataset


def load_class_weights(device: torch.device) -> torch.Tensor | None:
    p = C.DATA_DIR / "class_weights.json"
    if not p.exists() or not C.USE_CLASS_WEIGHTS:
        return None
    w = json.loads(p.read_text(encoding="utf-8"))
    return torch.tensor(w, dtype=torch.float, device=device)


def evaluate(model, loader, device, class_w):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    per_class_correct = torch.zeros(C.NUM_LABELS)
    per_class_total = torch.zeros(C.NUM_LABELS)
    autocast_dtype = torch.bfloat16 if C.USE_BF16 else torch.float32
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logits_fp32 = logits.float()
            loss = F.cross_entropy(logits_fp32, labels, weight=class_w)
            loss_sum += loss.item() * labels.size(0)
            preds = logits_fp32.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for c in range(C.NUM_LABELS):
                m = labels == c
                per_class_total[c] += m.sum().item()
                per_class_correct[c] += ((preds == c) & m).sum().item()
    return (
        loss_sum / max(total, 1),
        correct / max(total, 1),
        (per_class_correct / per_class_total.clamp(min=1)).tolist(),
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("[warn] CUDA not available — training on CPU will be very slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cuda={torch.cuda.is_available()} "
          f"name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")

    # A40 / Ampere — enable TF32 matmul (free speedup, fp32 path)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.BASE_MODEL,
        num_labels=C.NUM_LABELS,
        id2label=C.ID2LABEL,
        label2id=C.LABEL2ID,
    ).to(device)

    train_ds = CommentDataset(C.DATA_DIR / "train.jsonl", tokenizer, C.MAX_SEQ_LEN)
    val_ds = CommentDataset(C.DATA_DIR / "val.jsonl", tokenizer, C.MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=C.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=C.DATALOADER_NUM_WORKERS,
        pin_memory=C.PIN_MEMORY and device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.DATALOADER_NUM_WORKERS,
        pin_memory=C.PIN_MEMORY and device.type == "cuda",
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}  "
          f"steps/epoch={len(train_loader)}")

    class_w = load_class_weights(device)
    if class_w is not None:
        print("class weights:", class_w.tolist())

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * C.NUM_EPOCHS
    optimizer = AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * C.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    autocast_dtype = torch.bfloat16 if C.USE_BF16 else torch.float32
    use_autocast = C.USE_BF16 and device.type == "cuda"

    best_acc = -1.0
    best_path = C.MODEL_DIR / "best"
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_log: list[dict] = []

    for epoch in range(1, C.NUM_EPOCHS + 1):
        model.train()
        run_loss, seen = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True) if C.CONFIDENCE_AS_WEIGHT else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=use_autocast):
                logits = model(input_ids=ids, attention_mask=mask).logits
                per_ex = F.cross_entropy(
                    logits.float(),
                    labels,
                    weight=class_w,
                    label_smoothing=C.LABEL_SMOOTHING,
                    reduction="none",
                )
                loss = (per_ex * w).mean() if w is not None else per_ex.mean()
            loss.backward()
            if C.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % C.LOG_EVERY_N_STEPS == 0:
                print(f"  ep{epoch} step {step}/{steps_per_epoch} "
                      f"loss={run_loss / max(seen, 1):.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        epoch_secs = time.time() - t0
        val_loss, val_acc, per_class = evaluate(model, val_loader, device, class_w)
        print(f"[epoch {epoch}] {epoch_secs:.1f}s "
              f"train_loss={run_loss / max(seen, 1):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        for cid, name in C.ID2LABEL.items():
            print(f"  {name:18s} acc={per_class[cid]:.3f}")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": run_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class_acc": per_class,
            "epoch_secs": epoch_secs,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            (best_path / "training_config.json").write_text(
                json.dumps({
                    "base_model": C.BASE_MODEL,
                    "max_seq_len": C.MAX_SEQ_LEN,
                    "label2id": C.LABEL2ID,
                    "best_val_acc": best_acc,
                    "epoch": epoch,
                }, indent=2),
                encoding="utf-8",
            )
            print(f"  saved best → {best_path}")

    (C.LOG_DIR / "train_metrics.json").write_text(
        json.dumps(metrics_log, indent=2), encoding="utf-8"
    )
    print(f"\ndone. best val_acc={best_acc:.4f}  model={best_path}")


if __name__ == "__main__":
    main()
