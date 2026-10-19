# -*- coding: utf-8 -*-
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from Object_Localization_with_pytorch import (
    CANVAS_SIZE,
    EMOJI_DEFS,
    KNOWN_FILE_TO_CLASS,
    create_example_from_image,
    iou_per_sample,
    load_emoji_rgba,
    load_model,
    load_training_emojis,
    plot_bounding_box,
    emojis,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def collect_test_files(test_dir: Path):
    files = sorted(test_dir.glob("*.png"))
    return files


def run_evaluation(
    model,
    device,
    emoji_files: list[Path],
    per_emoji: int,
    seed: int,
    train_dir: Path | None = None,
):
    rng = np.random.default_rng(seed)
    records = []

    for path in emoji_files:
        stem = path.name
        known_class = KNOWN_FILE_TO_CLASS.get(stem)
        emoji_img = load_emoji_rgba(path)
        for _ in range(per_emoji):
            image, (row, col) = create_example_from_image(emoji_img, rng=rng)
            x = image / 255.0
            x_t = (
                torch.tensor(x, dtype=torch.float32)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                logits, box_pred = model(x_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_box = box_pred.cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            gt_box = np.array([row, col], dtype=np.float32)
            iou = float(
                iou_per_sample(
                    torch.tensor(gt_box).unsqueeze(0),
                    torch.tensor(pred_box).unsqueeze(0),
                )[0]
            )
            records.append(
                {
                    "file": stem,
                    "known_class": known_class,
                    "known_name": EMOJI_DEFS[known_class]["name"] if known_class is not None else None,
                    "pred_class": pred_class,
                    "pred_name": EMOJI_DEFS[pred_class]["name"],
                    "iou": iou,
                    "gt_box": gt_box.tolist(),
                    "pred_box": pred_box.tolist(),
                    "image": x,
                }
            )

    if train_dir:
        train_files = sorted(train_dir.glob("*.png"))
        for path in train_files:
            if path.parent.name == "test":
                continue
            stem = path.name
            known_class = KNOWN_FILE_TO_CLASS.get(stem)
            if known_class is None:
                continue
            emoji_img = load_emoji_rgba(path)
            for _ in range(per_emoji // 2):
                image, (row, col) = create_example_from_image(emoji_img, rng=rng)
                x = image / 255.0
                x_t = (
                    torch.tensor(x, dtype=torch.float32)
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    .to(device)
                )
                with torch.no_grad():
                    logits, box_pred = model(x_t)
                    pred_class = int(logits.argmax(dim=1).item())
                    pred_box = box_pred.cpu().numpy()[0]
                gt_box = np.array([row, col], dtype=np.float32)
                iou = float(
                    iou_per_sample(
                        torch.tensor(gt_box).unsqueeze(0),
                        torch.tensor(pred_box).unsqueeze(0),
                    )[0]
                )
                records.append(
                    {
                        "file": stem,
                        "known_class": known_class,
                        "known_name": EMOJI_DEFS[known_class]["name"],
                        "pred_class": pred_class,
                        "pred_name": EMOJI_DEFS[pred_class]["name"],
                        "iou": iou,
                        "split": "train_emoji",
                    }
                )

    return records


def summarize(records):
    ious = [r["iou"] for r in records if "split" not in r]
    known = [r for r in records if r.get("known_class") is not None and "split" not in r]
    unknown = [r for r in records if r.get("known_class") is None and "split" not in r]

    def stats(subset):
        if not subset:
            return {}
        vals = [r["iou"] for r in subset]
        return {
            "count": len(vals),
            "mean_iou": float(np.mean(vals)),
            "median_iou": float(np.median(vals)),
            "iou_ge_0.5": float(np.mean(np.array(vals) >= 0.5)),
            "iou_ge_0.75": float(np.mean(np.array(vals) >= 0.75)),
        }

    per_file = defaultdict(list)
    for r in records:
        if "split" in r:
            continue
        per_file[r["file"]].append(r["iou"])

    per_file_stats = {
        f: {
            "mean_iou": float(np.mean(v)),
            "known_class": KNOWN_FILE_TO_CLASS.get(f),
        }
        for f, v in per_file.items()
    }

    known_acc = None
    if known:
        known_acc = float(
            np.mean([r["pred_class"] == r["known_class"] for r in known])
        )

    unknown_pred_dist = {}
    if unknown:
        c = Counter(r["pred_name"] for r in unknown)
        unknown_pred_dist = dict(c)

    overall_subset = [{"iou": x} for x in ious]

    return {
        "overall": stats(overall_subset),
        "known_emoji": stats(known),
        "unknown_emoji": stats(unknown),
        "known_classification_accuracy": known_acc,
        "unknown_predicted_class_distribution": unknown_pred_dist,
        "per_file": per_file_stats,
    }


def plot_results(records, summary, out_dir: Path, run_label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    test_recs = [r for r in records if "split" not in r]
    ious = [r["iou"] for r in test_recs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ious, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Liczba próbek")
    ax.set_title(f"Rozkład IoU ({run_label})")
    fig.tight_layout()
    fig.savefig(out_dir / "iou_hist.png", dpi=150)
    plt.close(fig)

    per_file = summary["per_file"]
    files = sorted(per_file.keys())
    means = [per_file[f]["mean_iou"] for f in files]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(files, means, color="steelblue")
    ax.set_ylabel("Średnie IoU")
    ax.set_xlabel("Plik emoji")
    ax.set_title(f"IoU per plik testowy ({run_label})")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "iou_per_emoji.png", dpi=150)
    plt.close(fig)

    examples = sorted(test_recs, key=lambda r: r["iou"], reverse=True)[:3]
    examples += sorted(test_recs, key=lambda r: r["iou"])[:3]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, rec in zip(axes.flat, examples):
        vis = plot_bounding_box(
            rec["image"], rec["gt_box"], rec["pred_box"], norm=True
        )
        ax.imshow(vis)
        title = f"{rec['file']} IoU={rec['iou']:.2f}"
        if rec.get("known_name"):
            title += f"\nGT cls: {rec['known_name']}"
        title += f" pred: {rec['pred_name']}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Przykłady lokalizacji ({run_label})")
    fig.tight_layout()
    fig.savefig(out_dir / "examples.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / "models" / "emoji_detector_baseline.pth",
    )
    parser.add_argument("--test-dir", type=Path, default=SCRIPT_DIR / "emojis" / "test")
    parser.add_argument("--train-dir", type=Path, default=SCRIPT_DIR / "emojis")
    parser.add_argument("--per-emoji", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-label", type=str, default=None)
    args = parser.parse_args()

    load_training_emojis(args.train_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.model, device)
    run_label = args.run_label or args.model.stem

    test_files = collect_test_files(args.test_dir)
    records = run_evaluation(
        model,
        device,
        test_files,
        args.per_emoji,
        args.seed,
        train_dir=args.train_dir,
    )
    summary = summarize(records)
    summary["model_path"] = str(args.model)
    summary["val_iou_z_pliku"] = ckpt.get("val_iou")
    summary["epoka_z_pliku"] = ckpt.get("best_epoch")

    out_results = SCRIPT_DIR / "results" / f"test_{run_label}"
    out_figures = SCRIPT_DIR / "figures" / f"test_{run_label}"
    out_results.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in summary.items()}
    with open(out_results / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    plot_results(records, summary, out_figures, run_label)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
