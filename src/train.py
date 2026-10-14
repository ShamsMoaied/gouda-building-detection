"""
Step 3: fine-tune Mask R-CNN on the training chips.

After every epoch the model is scored on a separate validation tile (F1 at IoU 0.5).
The best epoch is kept, the learning rate drops when validation F1 stops improving,
and training stops when it hasn't improved for --patience epochs.
The test tile is not touched here.

Usage:
    python src/train.py --data data/chips_blocks/train --val data/chips_blocks/val --max-epochs 12
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from dataset import BuildingChips, collate
from evaluate import compute_metrics
from model import build_model


def keep_awake():
    """Ask Windows not to go to sleep while this process runs (my first run slowed 10x when it did)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/chips_blocks/train")
    parser.add_argument("--val", default="data/chips_blocks/val")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement")
    parser.add_argument("--batch", type=int, default=2)
    # torchvision's reference uses 0.02 for a batch of 16; scaled down to a batch of 2 that is 0.0025.
    # My first run with 0.01 went to NaN within 20 steps.
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup", type=int, default=200, help="steps of linear learning-rate warm-up")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    keep_awake()
    torch.manual_seed(42)
    ds = BuildingChips(args.data, augment=True)
    val_ds = BuildingChips(args.val)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         collate_fn=collate, num_workers=0)

    model = build_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # divide the learning rate by 10 when validation F1 hasn't improved for one epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=1)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    log = open(out / "train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "lr", "loss", "loss_classifier", "loss_box_reg", "loss_mask",
                     "loss_objectness", "loss_rpn_box_reg", "val_precision", "val_recall", "val_f1",
                     "val_pixel_iou", "minutes"])

    print(f"{len(ds)} training chips, {len(val_ds)} validation chips, {len(loader)} batches per epoch")
    best_f1, best_epoch, global_step = -1.0, 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        sums = {}
        for step, (images, targets, _) in enumerate(loader, start=1):
            # Warm-up: start at 1% of the learning rate and ramp up, so the new (random) heads
            # don't produce huge gradients in the first steps.
            global_step += 1
            if global_step <= args.warmup:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * (0.01 + 0.99 * global_step / args.warmup)

            losses = model(list(images), list(targets))
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at epoch {epoch} step {step}: {losses}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            if step % 50 == 0:
                rate = (time.time() - t0) / step
                print(f"  epoch {epoch} step {step}/{len(loader)}  loss {loss.item():.3f}  "
                      f"({rate:.1f} s/step)", flush=True)

        val, _ = compute_metrics(model, val_ds)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["f1"])

        n = len(loader)
        minutes = (time.time() - t0) / 60
        writer.writerow([epoch, lr] + [round(sums[k] / n, 4) for k in
                        ["loss", "loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness",
                         "loss_rpn_box_reg"]] +
                        [val["precision"], val["recall"], val["f1"], val["pixel_iou"], round(minutes, 1)])
        log.flush()
        torch.save(model.state_dict(), out / f"maskrcnn_gouda_epoch{epoch}.pth")

        improved = val["f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["f1"], epoch
            torch.save(model.state_dict(), out / "maskrcnn_gouda_best.pth")
            (out / "best.json").write_text(json.dumps({"epoch": epoch, "val": val}, indent=2))
        print(f"epoch {epoch}: loss {sums['loss'] / n:.3f} | val P {val['precision']:.3f} "
              f"R {val['recall']:.3f} F1 {val['f1']:.3f} pixIoU {val['pixel_iou']:.3f} | "
              f"lr {lr:.5f} | {minutes:.0f} min {'| new best' if improved else ''}", flush=True)

        if epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    log.close()
    print(f"best epoch {best_epoch} with validation F1 {best_f1:.3f} -> {out / 'maskrcnn_gouda_best.pth'}")


if __name__ == "__main__":
    main()
