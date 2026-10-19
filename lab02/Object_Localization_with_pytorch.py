# -*- coding: utf-8 -*-
"""Lokalizacja i klasyfikacja emoji (PyTorch) — rozszerzona wersja skryptu z ćwiczenia."""

import argparse
import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
CANVAS_SIZE = 144
EMOJI_SIZE = 72
BOX_SIZE = 52
BOX_OFFSET = 10

EMOJI_DEFS = {
    0: {"name": "happy", "file": "1F642.png"},
    1: {"name": "laughing", "file": "1F602.png"},
    2: {"name": "skeptical", "file": "1F928.png"},
    3: {"name": "sad", "file": "1F630.png"},
    4: {"name": "cool", "file": "1F60E.png"},
    5: {"name": "whoa", "file": "1F62F.png"},
    6: {"name": "crying", "file": "1F62D.png"},
    7: {"name": "puking", "file": "1F92E.png"},
    8: {"name": "nervous", "file": "1F62C.png"},
}

KNOWN_FILE_TO_CLASS = {v["file"]: k for k, v in EMOJI_DEFS.items()}

emojis: dict = {}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emoji_rgba(path: Path) -> Image.Image:
    png_file = Image.open(path).convert("RGBA")
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    return new_file


def load_training_emojis(emojis_dir: Path | None = None) -> dict:
    global emojis
    emojis_dir = emojis_dir or (SCRIPT_DIR / "emojis")
    loaded = {}
    for class_id, values in EMOJI_DEFS.items():
        img_path = emojis_dir / values["file"]
        loaded[class_id] = {
            "name": values["name"],
            "file": values["file"],
            "image": load_emoji_rgba(img_path),
        }
    emojis = loaded
    return loaded


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paste_emoji_on_canvas(emoji_image: Image.Image, row: int, col: int) -> np.ndarray:
    image = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    image[row : row + EMOJI_SIZE, col : col + EMOJI_SIZE, :] = np.array(emoji_image)
    return image


def bbox_from_placement(row: int, col: int) -> tuple[float, float]:
    return (row + BOX_OFFSET) / CANVAS_SIZE, (col + BOX_OFFSET) / CANVAS_SIZE


def create_example(class_id: int | None = None, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    if class_id is None:
        class_id = int(rng.integers(0, 9))
    row = int(rng.integers(0, CANVAS_SIZE - EMOJI_SIZE + 1))
    col = int(rng.integers(0, CANVAS_SIZE - EMOJI_SIZE + 1))
    image = paste_emoji_on_canvas(emojis[class_id]["image"], row, col)
    return image.astype("uint8"), class_id, bbox_from_placement(row, col)


def create_example_from_image(
    emoji_image: Image.Image, rng: np.random.Generator | None = None
):
    rng = rng or np.random.default_rng()
    row = int(rng.integers(0, CANVAS_SIZE - EMOJI_SIZE + 1))
    col = int(rng.integers(0, CANVAS_SIZE - EMOJI_SIZE + 1))
    image = paste_emoji_on_canvas(emoji_image, row, col)
    return image.astype("uint8"), bbox_from_placement(row, col)


def plot_bounding_box(image, gt_coords, pred_coords=(), norm=False):
    if norm:
        image = (image * 255.0).astype("uint8") if image.max() <= 1.0 else image.astype("uint8")

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    row, col = gt_coords
    row_px, col_px = int(row * CANVAS_SIZE), int(col * CANVAS_SIZE)
    draw.rectangle(
        (col_px, row_px, col_px + BOX_SIZE, row_px + BOX_SIZE), outline="green", width=3
    )

    if len(pred_coords) == 2:
        p_row, p_col = pred_coords
        p_row_px, p_col_px = int(p_row * CANVAS_SIZE), int(p_col * CANVAS_SIZE)
        draw.rectangle(
            (p_col_px, p_row_px, p_col_px + BOX_SIZE, p_row_px + BOX_SIZE),
            outline="red",
            width=3,
        )

    return np.array(image_pil)


def data_generator(batch_size=16, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    while True:
        x_batch = np.zeros((batch_size, CANVAS_SIZE, CANVAS_SIZE, 3))
        y_batch = np.zeros((batch_size,), dtype=np.int64)
        bbox_batch = np.zeros((batch_size, 2))

        for i in range(batch_size):
            image, class_id, bbox = create_example(rng=rng)
            x_batch[i] = image / 255.0
            y_batch[i] = class_id
            bbox_batch[i] = np.array(bbox)

        yield {"image": x_batch}, {"class_out": y_batch, "box_out": bbox_batch}


class EmojiDetector(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        in_channels = 3
        for i in range(5):
            out_channels = 2 ** (4 + i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(256 * 4 * 4, 256), nn.ReLU())
        self.class_out = nn.Linear(256, 9)
        self.box_out = nn.Linear(256, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.class_out(x), self.box_out(x)


def _box_corners(y):
    rows, cols = y[:, 0], y[:, 1]
    rows, cols = rows * CANVAS_SIZE, cols * CANVAS_SIZE
    y1, y2 = rows, rows + BOX_SIZE
    x1, x2 = cols, cols + BOX_SIZE
    return x1, y1, x2, y2


def iou_per_sample(y_true, y_pred):
    gt_x1, gt_y1, gt_x2, gt_y2 = _box_corners(y_true)
    p_x1, p_y1, p_x2, p_y2 = _box_corners(y_pred)

    i_x1 = torch.max(gt_x1, p_x1)
    i_y1 = torch.max(gt_y1, p_y1)
    i_x2 = torch.min(gt_x2, p_x2)
    i_y2 = torch.min(gt_y2, p_y2)

    i_w = torch.clamp(i_x2 - i_x1, min=0)
    i_h = torch.clamp(i_y2 - i_y1, min=0)
    i_area = i_w * i_h

    gt_area = torch.abs(gt_x2 - gt_x1) * torch.abs(gt_y2 - gt_y1)
    p_area = torch.abs(p_x2 - p_x1) * torch.abs(p_y2 - p_y1)
    u_area = gt_area + p_area - i_area

    return i_area / (u_area + 1e-7)


def calculate_iou(y_true, y_pred):
    return torch.mean(iou_per_sample(y_true, y_pred)).item()


def build_validation_set(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int64)
    boxes = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        image, class_id, bbox = create_example(rng=rng)
        images[i] = image / 255.0
        labels[i] = class_id
        boxes[i] = np.array(bbox)
    return images, labels, boxes


def evaluate(model, val_images, val_labels, val_boxes, device, box_weight: float):
    model.eval()
    class_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss()

    x = torch.tensor(val_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    y_cls = torch.tensor(val_labels, dtype=torch.long).to(device)
    y_box = torch.tensor(val_boxes, dtype=torch.float32).to(device)

    with torch.no_grad():
        class_logits, box_preds = model(x)
        loss_class = class_criterion(class_logits, y_cls)
        loss_box = box_criterion(box_preds, y_box)
        total_loss = loss_class + box_weight * loss_box
        iou = calculate_iou(y_box, box_preds)
        acc = (class_logits.argmax(dim=1) == y_cls).float().mean().item()

    return {
        "class_loss": loss_class.item(),
        "box_loss": loss_box.item(),
        "total_loss": total_loss.item(),
        "iou": iou,
        "accuracy": acc,
    }


def plot_cost_landscape(out_path: Path):
    gt_row, gt_col = 40 / CANVAS_SIZE, 50 / CANVAS_SIZE
    shifts = np.linspace(-80, 80, 161)
    mse_vals, iou_vals, giou_vals = [], [], []

    def numpy_iou_giou(pr, pc):
        gr, gc = gt_row * CANVAS_SIZE, gt_col * CANVAS_SIZE
        pr_px, pc_px = pr * CANVAS_SIZE, pc * CANVAS_SIZE
        gt = (gc, gr, gc + BOX_SIZE, gr + BOX_SIZE)
        pred = (pc_px, pr_px, pc_px + BOX_SIZE, pr_px + BOX_SIZE)
        ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
        ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_gt = BOX_SIZE * BOX_SIZE
        area_p = BOX_SIZE * BOX_SIZE
        union = area_gt + area_p - inter
        iou = inter / (union + 1e-7)
        cx1 = min(gt[0], pred[0])
        cy1 = min(gt[1], pred[1])
        cx2 = max(gt[2], pred[2])
        cy2 = max(gt[3], pred[3])
        c_area = (cx2 - cx1) * (cy2 - cy1)
        giou = iou - (c_area - union) / (c_area + 1e-7)
        return iou, giou

    for d in shifts:
        pr = gt_row + d / CANVAS_SIZE
        pc = gt_col
        mse_vals.append((pr - gt_row) ** 2 + (pc - gt_col) ** 2)
        iou, giou = numpy_iou_giou(pr, pc)
        iou_vals.append(iou)
        giou_vals.append(giou)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(shifts, mse_vals, label="MSE (norm. coords)")
    axes[0].plot(shifts, 1 - np.array(iou_vals), label="1 - IoU")
    axes[0].plot(shifts, 1 - np.array(giou_vals), label="1 - GIoU")
    axes[0].set_xlabel("Przesunięcie predykcji w osi row [px]")
    axes[0].set_ylabel("Koszt")
    axes[0].legend()
    axes[0].set_title("Koszt w funkcji przesunięcia (1D)")
    axes[0].grid(True, alpha=0.3)

    grid = np.linspace(-60, 60, 121)
    heat = np.zeros((len(grid), len(grid)))
    for i, dr in enumerate(grid):
        for j, dc in enumerate(grid):
            pr = gt_row + dr / CANVAS_SIZE
            pc = gt_col + dc / CANVAS_SIZE
            _, giou = numpy_iou_giou(pr, pc)
            heat[i, j] = 1 - giou

    im = axes[1].imshow(
        heat,
        origin="lower",
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("Przesunięcie col [px]")
    axes[1].set_ylabel("Przesunięcie row [px]")
    axes[1].set_title("1 - GIoU (mapa 2D)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_history(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history["step_losses"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    ax.plot([s["class_loss"] for s in steps], alpha=0.4, label="CE (krok)")
    ax.plot([s["box_loss"] for s in steps], alpha=0.4, label="MSE box (krok)")
    ax.plot([s["total_loss"] for s in steps], alpha=0.4, label="Suma (krok)")
    epochs = history["epochs"]
    ep = np.arange(1, len(epochs) + 1)
    ax.plot(ep, [e["train_class_loss"] for e in epochs], "o-", label="CE (epoka)")
    ax.plot(ep, [e["train_box_loss"] for e in epochs], "o-", label="MSE box (epoka)")
    ax.plot(ep, [e["train_total_loss"] for e in epochs], "o-", label="Suma (epoka)")
    ax.plot(ep, [e["val_total_loss"] for e in epochs], "s--", label="Suma wal.")
    ax.set_ylabel("Koszt")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Funkcje kosztu")

    ax2 = axes[1]
    ax2.plot(ep, [e["train_iou"] for e in epochs], "o-", label="IoU trening")
    ax2.plot(ep, [e["val_iou"] for e in epochs], "s--", label="IoU walidacja")
    ax2.plot(ep, [e["val_accuracy"] for e in epochs], "^-", label="Accuracy wal.")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Metryka")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("IoU i klasyfikacja")

    fig.tight_layout()
    fig.savefig(out_dir / "losses.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, [e["train_iou"] for e in epochs], "o-", label="IoU trening")
    ax.plot(ep, [e["val_iou"] for e in epochs], "s--", label="IoU walidacja")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iou_acc.png", dpi=150)
    plt.close(fig)


def save_model(model, path: Path, best_epoch: int, best_val_iou: float, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": [EMOJI_DEFS[i]["name"] for i in range(9)],
            "config": config,
            "best_epoch": best_epoch,
            "val_iou": float(best_val_iou),
        },
        path,
    )


def load_model(path: Path, device=None):
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = EmojiDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def test_model_panel(model, test_datagen, device):
    model.eval()
    example, label = next(test_datagen)
    x = example["image"]
    y = label["class_out"]
    box = label["box_out"]

    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        pred_y_logits, pred_box_tensor = model(x_tensor)
        pred_y = F.softmax(pred_y_logits, dim=1).cpu().numpy()
        pred_box = pred_box_tensor.cpu().numpy()

    pred_coords = pred_box[0]
    gt_coords = box[0]
    pred_class = int(np.argmax(pred_y[0]))
    gt_class = int(y[0])
    image = x[0]
    iou = float(iou_per_sample(
        torch.tensor(box, dtype=torch.float32),
        torch.tensor(pred_box, dtype=torch.float32),
    )[0])

    image = plot_bounding_box(image, gt_coords, pred_coords, norm=True)
    gt_name = emojis[gt_class]["name"]
    pred_name = emojis[pred_class]["name"]
    color = "green" if gt_class == pred_class else "red"

    plt.imshow(image)
    plt.xlabel(f"Pred: {pred_name}", color=color)
    plt.ylabel(f"GT: {gt_name}", color=color)
    plt.title(f"IoU={iou:.2f}")
    plt.xticks([])
    plt.yticks([])
    return iou


def save_test_grid(model, out_path: Path, device, seed: int):
    rng = np.random.default_rng(seed)
    gen = data_generator(1, rng=rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 4))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        test_model_panel(model, gen, device)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=64, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-samples-each-epoch", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Urządzenie: {device}")

    load_training_emojis()

    figures_dir = SCRIPT_DIR / "figures" / args.run_name
    results_dir = SCRIPT_DIR / "results" / args.run_name
    models_dir = SCRIPT_DIR / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    plot_cost_landscape(SCRIPT_DIR / "figures" / "cost_landscape.png")

    val_images, val_labels, val_boxes = build_validation_set(args.val_samples, args.seed + 1)

    model = EmojiDetector().to(device)
    print(f"Parametry: {count_parameters(model):,}")

    class_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    train_rng = np.random.default_rng(args.seed + 2)
    train_gen = data_generator(batch_size=args.batch_size, rng=train_rng)

    history = {"step_losses": [], "epochs": [], "config": vars(args)}
    best_val_iou = -1.0
    best_epoch = 0
    patience_counter = 0
    model_path = models_dir / f"emoji_detector_{args.run_name}.pth"

    if not args.no_samples_each_epoch:
        save_test_grid(
            model,
            figures_dir / "samples" / "epoch_00_untrained.png",
            device,
            args.seed + 100,
        )

    print("\nTrening...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_class_loss = 0.0
        epoch_box_loss = 0.0
        epoch_iou = 0.0

        for _ in range(args.steps):
            inputs, targets = next(train_gen)
            x_train = (
                torch.tensor(inputs["image"], dtype=torch.float32)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            y_class = torch.tensor(targets["class_out"], dtype=torch.long).to(device)
            y_box = torch.tensor(targets["box_out"], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            class_logits, box_preds = model(x_train)
            loss_class = class_criterion(class_logits, y_class)
            loss_box = box_criterion(box_preds, y_box)
            total_loss = loss_class + args.box_weight * loss_box
            total_loss.backward()
            optimizer.step()

            step_iou = calculate_iou(y_box, box_preds)
            history["step_losses"].append(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "class_loss": loss_class.item(),
                    "box_loss": loss_box.item(),
                    "total_loss": total_loss.item(),
                    "iou": step_iou,
                }
            )
            global_step += 1

            epoch_class_loss += loss_class.item()
            epoch_box_loss += loss_box.item()
            epoch_iou += step_iou

        val_metrics = evaluate(
            model, val_images, val_labels, val_boxes, device, args.box_weight
        )
        avg_class = epoch_class_loss / args.steps
        avg_box = epoch_box_loss / args.steps
        avg_iou = epoch_iou / args.steps
        lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_class_loss": avg_class,
            "train_box_loss": avg_box,
            "train_total_loss": avg_class + args.box_weight * avg_box,
            "train_iou": avg_iou,
            "val_class_loss": val_metrics["class_loss"],
            "val_box_loss": val_metrics["box_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_iou": val_metrics["iou"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": lr,
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch + 1}/{args.epochs} -> "
            f"CE: {avg_class:.4f} | MSE: {avg_box:.4f} | IoU train: {avg_iou:.4f} | "
            f"IoU val: {val_metrics['iou']:.4f} | acc val: {val_metrics['accuracy']:.4f} | LR: {lr:.7f}"
        )

        plot_training_history(history, figures_dir)
        with open(results_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if not args.no_samples_each_epoch:
            save_test_grid(
                model,
                figures_dir / "samples" / f"epoch_{epoch + 1:02d}.png",
                device,
                args.seed + 200 + epoch,
            )

        scheduler.step()
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], 3e-7)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch + 1
            patience_counter = 0
            save_model(
                model,
                model_path,
                best_epoch,
                best_val_iou,
                vars(args),
            )
            print(f"  zapis modelu (IoU wal.={best_val_iou:.4f}) -> {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop w epoce {epoch + 1}, najlepsze IoU wal.: {best_val_iou:.4f}")
                break

    print(f"\nKoniec. Epoka {best_epoch}, IoU wal. {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
