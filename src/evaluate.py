"""
Step 4: evaluate a trained model on held-out chips.

A prediction is a true positive when it overlaps a not-yet-matched label with IoU >= 0.5.
Matching is greedy, highest score first (the usual COCO-style approach, at a single threshold).

Usage:
    python src/evaluate.py                                   # test tile, BAG buildings as labels
    python src/evaluate.py --data data/chips_blocks/test     # test tile, merged blocks as labels
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BuildingChips
from model import build_model

PIXEL_AREA = 0.25 * 0.25   # m2 per pixel
SIZE_BINS = [(0, 50), (50, 150), (150, 1e9)]   # m2: sheds/garages, houses, large buildings


def match(pred_masks, scores, gt_masks, iou_thr):
    """Return list of (pred_idx, gt_idx, iou) for matched pairs."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return []
    p = pred_masks.reshape(len(pred_masks), -1).astype(np.float32)
    g = gt_masks.reshape(len(gt_masks), -1).astype(np.float32)
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)

    pairs, used = [], set()
    for pi in np.argsort(-scores):
        candidates = [(iou[pi, gi], gi) for gi in range(len(g)) if gi not in used]
        if not candidates:
            break
        best_iou, gi = max(candidates)
        if best_iou >= iou_thr:
            pairs.append((pi, gi, float(best_iou)))
            used.add(gi)
    return pairs


@torch.no_grad()
def compute_metrics(model, ds, score_thr=0.5, iou_thr=0.5, keep_examples=False):
    model.eval()
    tp = fp = fn = 0
    ious = []
    inter_px = union_px = 0
    size_found = {b: [0, 0] for b in SIZE_BINS}   # bin -> [found, total]
    examples = []

    for i in range(len(ds)):
        img, target, name = ds[i]
        out = model([img])[0]
        keep = out["scores"] >= score_thr
        pred = (out["masks"][keep, 0] > 0.5).numpy()
        scores = out["scores"][keep].numpy()
        gt = target["masks"].numpy().astype(bool)

        pairs = match(pred, scores, gt, iou_thr)
        tp += len(pairs)
        fp += len(pred) - len(pairs)
        fn += len(gt) - len(pairs)
        ious += [p[2] for p in pairs]

        matched_gt = {p[1] for p in pairs}
        for gi, m in enumerate(gt):
            area = m.sum() * PIXEL_AREA
            for b in SIZE_BINS:
                if b[0] <= area < b[1]:
                    size_found[b][1] += 1
                    size_found[b][0] += gi in matched_gt

        # pixel-level IoU: "is this pixel part of any building?"
        pred_any = pred.any(0) if len(pred) else np.zeros(img.shape[1:], bool)
        gt_any = gt.any(0) if len(gt) else np.zeros(img.shape[1:], bool)
        inter_px += (pred_any & gt_any).sum()
        union_px += (pred_any | gt_any).sum()

        if keep_examples and len(gt) >= 15:
            examples.append((name, img, gt, pred))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    metrics = {
        "chips": len(ds),
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_iou_of_matches": round(float(np.mean(ious)), 3) if ious else None,
        "pixel_iou": round(float(inter_px / max(union_px, 1)), 3),
        "recall_by_size_m2": {f"{lo}-{hi if hi < 1e9 else 'max'}": {
            "found": int(f), "total": t, "recall": round(f / max(t, 1), 3)}
            for (lo, hi), (f, t) in size_found.items()},
    }
    return metrics, examples


def plot_examples(examples, out_path):
    """Side-by-side: labels vs predictions, for three busy chips."""
    examples = examples[:: max(1, len(examples) // 3)][:3]
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(9, 4.5 * len(examples)))
    for row, (name, img, gt, pred) in zip(np.atleast_2d(axes), examples):
        rgb = img.permute(1, 2, 0).numpy()
        for ax, masks, title, colour in ((row[0], gt, "Label", "yellow"),
                                         (row[1], pred, "Model prediction", "cyan")):
            ax.imshow(rgb)
            for m in masks:
                ax.contour(m, levels=[0.5], colors=colour, linewidths=0.8)
            ax.set_title(f"{title}: {len(masks)} objects", fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/chips/test")
    parser.add_argument("--weights", default="models/maskrcnn_gouda_best.pth")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--name", default="test", help="suffix for the output files in docs/")
    args = parser.parse_args()

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    metrics, examples = compute_metrics(model, BuildingChips(args.data), args.score, args.iou,
                                        keep_examples=True)
    metrics = {"data": args.data, "weights": args.weights, **metrics}
    print(json.dumps(metrics, indent=2))
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/metrics_{args.name}.json").write_text(json.dumps(metrics, indent=2))
    plot_examples(examples, f"docs/predictions_{args.name}.png")
    print(f"saved docs/metrics_{args.name}.json and docs/predictions_{args.name}.png")


if __name__ == "__main__":
    main()
