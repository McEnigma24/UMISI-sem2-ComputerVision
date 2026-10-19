# -*- coding: utf-8 -*-

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
IMGS_DIR = SCRIPT_DIR / "out_" / "imgs"
DEPTH16_DIR = SCRIPT_DIR / "out_" / "depth16"

HEAD_IDX = slice(0, 5)
ARM_IDX = slice(5, 11)
HIP_IDX = slice(11, 13)
LEG_IDX = slice(13, 17)

def load_manual_labels(path: Path) -> dict[int, dict]:
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            labels[idx] = {
                "person": int(row["person_visible"]),
                "body": row["body_part"],
                "blur": int(row["blurred"]),
            }
    return labels


def list_frames():
    return sorted(IMGS_DIR.glob("*_img.png"))


def frame_index(path: Path) -> int:
    return int(path.stem.split("_")[0])


def laplacian_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_depth16(frame_idx: int) -> np.ndarray | None:
    path = DEPTH16_DIR / f"{frame_idx:04d}_depth.png"
    if not path.exists():
        return None
    depth = np.array(Image.open(path))
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return depth


def depth_at_keypoints(depth: np.ndarray, kpts: np.ndarray, conf_thr=0.5):
    if depth is None or kpts is None or len(kpts) == 0:
        return None
    h, w = depth.shape[:2]
    vals = []
    for i in range(len(kpts)):
        if i in range(11, 17) and kpts[i, 2] >= conf_thr:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            if 0 <= x < w and 0 <= y < h:
                d = int(depth[y, x])
                if d > 0:
                    vals.append(d)
    if not vals and len(kpts) > 0:
        xs = kpts[:, 0]
        ys = kpts[:, 1]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        if 0 <= cx < w and 0 <= cy < h:
            d = int(depth[cy, cx])
            if d > 0:
                vals.append(d)
    if not vals:
        return None
    return float(np.median(vals))


def visible_groups(kpts, conf_thr=0.5):
    if kpts is None or len(kpts) == 0:
        return {g: 0 for g in ("head", "arms", "hips", "legs")}
    out = {}
    for name, sl in (
        ("head", HEAD_IDX),
        ("arms", ARM_IDX),
        ("hips", HIP_IDX),
        ("legs", LEG_IDX),
    ):
        seg = kpts[sl]
        out[name] = int(np.sum(seg[:, 2] >= conf_thr))
    return out


def resolve_model_names():
    candidates = [
        ("n", "yolo26n-pose.pt"),
        ("s", "yolo26s-pose.pt"),
        ("m", "yolo26m-pose.pt"),
        ("n", "yolo11n-pose.pt"),
        ("s", "yolo11s-pose.pt"),
        ("m", "yolo11m-pose.pt"),
    ]
    chosen = {}
    for size, name in candidates:
        if size not in chosen:
            chosen[size] = name
    return chosen


def run_model_on_dataset(model_name: str, device: str, conf: float = 0.25):
    yolo_dir = SCRIPT_DIR / "models" / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    frames = list_frames()
    records = []

    for path in frames:
        idx = frame_index(path)
        bgr = cv2.imread(str(path))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = laplacian_blur(gray)
        depth = read_depth16(idx)

        t0 = time.perf_counter()
        results = model.predict(path, device=device, conf=conf, verbose=False)
        dt = time.perf_counter() - t0

        best_person = None
        best_conf = 0.0
        n_persons = 0
        if results and results[0].boxes is not None:
            n_persons = len(results[0].boxes)
            for i in range(n_persons):
                cls = int(results[0].boxes.cls[i].item())
                if cls != 0:
                    continue
                c = float(results[0].boxes.conf[i].item())
                if c > best_conf:
                    best_conf = c
                    kpts = None
                    if results[0].keypoints is not None and i < len(results[0].keypoints.data):
                        kpts = results[0].keypoints.data[i].cpu().numpy()
                    best_person = kpts

        groups = visible_groups(best_person)
        n_kpts_vis = sum(groups.values())
        dist_mm = depth_at_keypoints(depth, best_person) if best_person is not None else None

        records.append(
            {
                "frame_idx": idx,
                "file": path.name,
                "n_persons": n_persons,
                "best_conf": best_conf,
                "n_keypoints_visible": n_kpts_vis,
                "groups": groups,
                "depth_mm": dist_mm,
                "blur_laplacian": blur,
                "inference_s": dt,
                "detected": int(best_conf >= conf),
            }
        )
    return records


def benchmark_speed(model_name: str, n_warmup=3, n_iter=30):
    frames = list_frames()[:50]
    if not frames:
        return {}
    model = YOLO(model_name)
    out = {}
    for device in ("cpu", "0"):
        for _ in range(n_warmup):
            model.predict(frames[0], device=device, verbose=False)
        times = []
        for path in frames[:n_iter]:
            t0 = time.perf_counter()
            model.predict(path, device=device, verbose=False)
            times.append(time.perf_counter() - t0)
        out[device] = {
            "mean_ms": float(np.mean(times) * 1000),
            "fps": float(1.0 / np.mean(times)),
        }
    return out


def detection_metrics_vs_manual(model_records: dict, manual_labels: dict):
    metrics = {}
    for model_key, recs in model_records.items():
        by_idx = {r["frame_idx"]: r for r in recs}
        tp = fp = fn = tn = 0
        for idx, ann in manual_labels.items():
            pred = by_idx.get(idx, {}).get("detected", 0)
            gt = ann["person"]
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        metrics[model_key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        }
    return metrics


def plot_time_series(all_records: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for model_key, recs in all_records.items():
        x = [r["frame_idx"] for r in recs]
        axes[0, 0].plot(x, [r["best_conf"] for r in recs], alpha=0.7, label=model_key)
        axes[0, 1].plot(x, [r["n_keypoints_visible"] for r in recs], alpha=0.7, label=model_key)
        depths = [r["depth_mm"] if r["depth_mm"] else np.nan for r in recs]
        axes[1, 0].plot(x, depths, alpha=0.7, label=model_key)
        axes[1, 1].plot(x, [r["blur_laplacian"] for r in recs], alpha=0.7, label=model_key)
    axes[0, 0].set_title("Pewność detekcji osoby")
    axes[0, 1].set_title("Liczba widocznych stawów (conf>=0.5)")
    axes[1, 0].set_title("Odległość z depth16 [mm]")
    axes[1, 1].set_title("Wariancja Laplasjanu (rozmycie)")
    for ax in axes.flat:
        ax.set_xlabel("Indeks klatki")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_series.png", dpi=150)
    plt.close(fig)


def plot_joint_visibility(all_records: dict, out_dir: Path):
    model_key = list(all_records.keys())[-1]
    recs = all_records[model_key]
    sums = {g: 0 for g in ("head", "arms", "hips", "legs")}
    n = 0
    for r in recs:
        if r["best_conf"] < 0.25:
            continue
        n += 1
        for g, v in r["groups"].items():
            sums[g] += v
    if n == 0:
        return
    labels = list(sums.keys())
    vals = [sums[l] / n for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color="teal")
    ax.set_ylabel("Średnia liczba widocznych stawów w grupie")
    ax.set_title(f"Widoczność grup stawów ({model_key}, conf>=0.25)")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_groups.png", dpi=150)
    plt.close(fig)


def plot_speed_comparison(speed: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = list(speed.keys())
    cpu_fps = [speed[m]["cpu"]["fps"] for m in models]
    gpu_fps = [speed[m]["0"]["fps"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, cpu_fps, w, label="CPU")
    ax.bar(x + w / 2, gpu_fps, w, label="GPU")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("FPS")
    ax.set_title("Porównanie szybkości inferencji")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "speed_comparison.png", dpi=150)
    plt.close(fig)


def save_example_montage(model_name: str, out_path: Path):
    model = YOLO(model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks = [0, 150, 200, 230, 300, 350, 100, 390]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, idx in zip(axes.flat, picks):
        path = IMGS_DIR / f"{idx:04d}_img.png"
        if not path.exists():
            ax.axis("off")
            continue
        res = model.predict(path, device=0, conf=0.25, verbose=False)[0]
        plotted = res.plot()
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        ax.imshow(plotted)
        ax.set_title(f"{idx:04d}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"YOLO-Pose — przykładowe klatki ({model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def create_review_montage(manual_labels: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(manual_labels.keys())
    cols = 8
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.2))
    axes = np.atleast_2d(axes)
    for ax, idx in zip(axes.flat, indices):
        path = IMGS_DIR / f"{idx:04d}_img.png"
        img = np.array(Image.open(path).resize((212, 120)))
        ax.imshow(img)
        ann = manual_labels[idx]
        ax.set_title(
            f"{idx} p={ann['person']} {ann['body']} b={ann['blur']}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes.flat[len(indices) :]:
        ax.axis("off")
    fig.suptitle("Podzbiór klatek (co 10.) — etykiety ręczne")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    results_dir = SCRIPT_DIR / "results" / "yolo"
    figures_dir = SCRIPT_DIR / "figures" / "yolo"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels_path = results_dir / "manual_labels.csv"
    manual_labels = load_manual_labels(labels_path)
    create_review_montage(manual_labels, figures_dir / "manual_review_grid.png")

    model_map = resolve_model_names()
    all_records = {}
    speed = {}

    for size, model_name in model_map.items():
        print(f"{model_name} — cały zbiór (GPU)...")
        recs = run_model_on_dataset(model_name, device="0", conf=args.conf)
        all_records[size] = recs
        with open(results_dir / f"metrics_{size}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f)

        print(f"Pomiar czasu: {model_name}")
        speed[size] = benchmark_speed(model_name)

    det_metrics = detection_metrics_vs_manual(all_records, manual_labels)
    with open(results_dir / "detection_vs_manual.json", "w", encoding="utf-8") as f:
        json.dump(det_metrics, f, indent=2)
    with open(results_dir / "speed_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(speed, f, indent=2)

    plot_time_series(all_records, figures_dir)
    plot_joint_visibility(all_records, figures_dir)
    plot_speed_comparison(speed, figures_dir)
    save_example_montage(model_map["m"], figures_dir / "examples_montage.png")

    print("Detekcja vs etykiety ręczne:", json.dumps(det_metrics, indent=2))
    print("FPS:", json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
