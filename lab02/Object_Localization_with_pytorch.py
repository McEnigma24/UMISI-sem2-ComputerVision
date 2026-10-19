# -*- coding: utf-8 -*-
"""
Multi-Output Object Localization and Classification with PyTorch
Dataset: Real Emoji PNG Images with Transparent Backgrounds
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# OpenMP runtime duplication error (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 64
EPOCH_SIZE = 64

# ==========================================
# DATASET CONFIGURATION & PREPROCESSING
# ==========================================

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

# Pre-load and convert emoji PNG files to RGB with a solid white background
EMOJIS_DIR = 'emojis'
try:
    for class_id, values in emojis.items():
        img_path = os.path.join(EMOJIS_DIR, values['file'])
        png_file = Image.open(img_path).convert('RGBA')
        png_file.load()
        
        # Create a new white canvas of the same size
        new_file = Image.new("RGB", png_file.size, (255, 255, 255))
        # Paste the emoji using its alpha channel as a mask
        new_file.paste(png_file, mask=png_file.split()[3])
        emojis[class_id]['image'] = new_file
    print("Successfully loaded all emoji source files into memory.")
except FileNotFoundError:
    print(f"\n[CRITICAL ERROR] The directory '{EMOJIS_DIR}' or emoji PNG files were not found.")
    print("Please ensure your working directory contains the 'emojis' folder with files like '1F642.png'.")
    print("Creating mock fallback images so the execution can proceed seamlessly for demonstration purposes...")
    # Fallback placeholder images if directory is missing during laboratory test
    for class_id, values in emojis.items():
        mock_img = Image.new("RGB", (72, 72), (np.random.randint(50,200), np.random.randint(50,200), np.random.randint(50,200)))
        emojis[class_id]['image'] = mock_img

# ==========================================
# HELPER FUNCTIONS & DATA GENERATOR
# ==========================================

def create_example():
    """
    Generates a single training example: a 144x144 white canvas with a random emoji pasted.
    Returns: normalized image array, class ID, and normalized bounding box row/col coordinates.
    """
    class_id = np.random.randint(0, 9)
    image = np.ones((144, 144, 3), dtype=np.uint8) * 255  # White background
    row = np.random.randint(0, 72)
    col = np.random.randint(0, 72)
    
    # Paste the preprocessed PIL emoji image into the random sub-region grid coordinates
    image[row: row + 72, col: col + 72, :] = np.array(emojis[class_id]['image'])
    
    # Coordinates adjustment matches original script normalization rules
    return image.astype('uint8'), class_id, (row + 10) / 144, (col + 10) / 144

def plot_bounding_box(image, gt_coords, pred_coords=[], norm=False):
    """
    Draws ground truth (green) and prediction (red) boxes onto the specified image.
    """
    if norm:
        image = (image * 255.).astype('uint8') if image.max() <= 1.0 else image.astype('uint8')
        
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    # Draw Green Ground Truth Box
    row, col = gt_coords
    row_px, col_px = int(row * 144), int(col * 144)
    draw.rectangle((col_px, row_px, col_px + 52, row_px + 52), outline='green', width=3)

    # Draw Red Prediction Box
    if len(pred_coords) == 2:
        p_row, p_col = pred_coords
        p_row_px, p_col_px = int(p_row * 144), int(p_col * 144)
        draw.rectangle((p_col_px, p_row_px, p_col_px + 52, p_row_px + 52), outline='red', width=3)
        
    return np.array(image_pil)

def data_generator(batch_size=16):
    while True:
        x_batch = np.zeros((batch_size, 144, 144, 3))
        y_batch = np.zeros((batch_size, 9))
        bbox_batch = np.zeros((batch_size, 2))

        for i in range(0, batch_size):
            image, class_id, row, col = create_example()
            x_batch[i] = image / 255.0  # Normalize pixel intensities to [0.0, 1.0]
            y_batch[i, class_id] = 1.0  # One-hot encoded labels
            bbox_batch[i] = np.array([row, col])
            
        yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}

# Verification step - Plot a generated dataset sample
gen = data_generator(1)
example, label = next(gen)
image_check = example['image'][0]
class_id_check = np.argmax(label['class_out'][0])
coords_check = label['box_out'][0]

image_check = plot_bounding_box(image_check, coords_check, norm=True)
plt.imshow(image_check)
plt.title(f"Dataset Verification: {emojis[class_id_check]['name']}")
plt.axis('off')
plt.show()

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class EmojiDetector(nn.Module):
    def __init__(self):
        super(EmojiDetector, self).__init__()
        
        layers = []
        in_channels = 3
        for i in range(0, 5):
            out_channels = 2**(4 + i)  # 16, 32, 64, 128, 256
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU()
        )
        
        self.class_out = nn.Linear(256, 9)
        self.box_out = nn.Linear(256, 2)
        
        # WYWOŁANIE INICJALIZACJI WAG
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Keras Conv2D i Linear uzywaja domyslnie Xavier Uniform
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        class_logits = self.class_out(x)
        box_preds = self.box_out(x)
        
        return class_logits, box_preds


model = EmojiDetector().to(device)

# ==========================================
# CRITERIA & METRIC DEFINITIONS
# ==========================================

class_criterion = nn.CrossEntropyLoss()
box_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

def calculate_iou(y_true, y_pred):
    """
    Computes batch-wise IoU (Intersection over Union) using PyTorch tensors.
    """
    def get_box(y):
        rows, cols = y[:, 0], y[:, 1]
        rows, cols = rows * 144, cols * 144
        y1, y2 = rows, rows + 52
        x1, x2 = cols, cols + 52
        return x1, y1, x2, y2

    gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
    p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

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

    iou = i_area / (u_area + 1e-7)
    return torch.mean(iou).item()

# ==========================================
#  TEST & CALLBACK ENGINE
# ==========================================

def test_model(model, test_datagen):
    model.eval()
    example, label = next(test_datagen)
    x = example['image']
    y = label['class_out']
    box = label['box_out']

    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        pred_y_logits, pred_box_tensor = model(x_tensor)
        pred_y = F.softmax(pred_y_logits, dim=1).cpu().numpy()
        pred_box = pred_box_tensor.cpu().numpy()

    pred_coords = pred_box[0]
    gt_coords = box[0]
    pred_class = np.argmax(pred_y[0])
    image = x[0]

    gt = emojis[np.argmax(y[0])]['name']
    pred_class_name = emojis[pred_class]['name']

    image = plot_bounding_box(image, gt_coords, pred_coords, norm=True)
    color = 'green' if gt == pred_class_name else 'red'

    plt.imshow(image)
    plt.xlabel(f'Pred: {pred_class_name}', color=color)
    plt.ylabel(f'GT: {gt}', color=color)
    plt.xticks([])
    plt.yticks([])

def test(model):
    test_datagen = data_generator(1)
    plt.figure(figsize=(16, 4))
    for i in range(0, 6):
        plt.subplot(1, 6, i + 1)
        test_model(model, test_datagen)
    plt.show()

print('Evaluating untrained baseline visualization...')
test(model)

# ==========================================
# PIPELINE TRAINING LOOP
# ==========================================

train_gen = data_generator(batch_size=BATCH_SIZE)
epochs = 10
steps_per_epoch = EPOCH_SIZE

best_iou = -1.0
patience = 3
patience_counter = 0

print("\nStarting the network training execution pipeline...")

for epoch in range(epochs):
    model.train()
    epoch_class_loss = 0.0
    epoch_box_loss = 0.0
    epoch_iou = 0.0
    
    for step in range(steps_per_epoch):
        inputs, targets = next(train_gen)
        
        # Transpose shapes from (B, H, W, C) to PyTorch layout requirement (B, C, H, W)
        x_train = torch.tensor(inputs['image'], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        y_class = torch.tensor(targets['class_out'], dtype=torch.float32).to(device)
        y_box = torch.tensor(targets['box_out'], dtype=torch.float32).to(device)
        

        # --- Internal Step Loop Level (4 spaces indentation inside 'for step in range(...)') ---
        optimizer.zero_grad()
        
        # Forward pass and loss tracking
        class_logits, box_preds = model(x_train)
        loss_class = class_criterion(class_logits, y_class)
        loss_box = box_criterion(box_preds, y_box)
        total_loss = loss_class + loss_box
        #total_loss = loss_class + (10.0 * loss_box)
        
        # Optimization step
        total_loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        epoch_class_loss += loss_class.item()
        epoch_box_loss += loss_box.item()
        epoch_iou += calculate_iou(y_box, box_preds)

    # --- Main Epoch Loop Level (no extra step indentation inside 'for epoch in range(...)') ---
    avg_class_loss = epoch_class_loss / steps_per_epoch
    avg_box_loss = epoch_box_loss / steps_per_epoch
    avg_iou = epoch_iou / steps_per_epoch
    
    # Track parameter metrics
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} -> Class Loss: {avg_class_loss:.4f} | Box MSE Loss: {avg_box_loss:.4f} | Box IoU: {avg_iou:.4f} | LR: {current_lr:.7f}")
    
    # Draw performance samples on screen at the end of each training loop epoch
    test(model)
    
    # Learning rate adjustments
    scheduler.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], 3e-7)
        
    # Early stopping verification logic tracker
    if avg_iou > best_iou:
        best_iou = avg_iou
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best achieved IoU score: {best_iou:.4f}")
            break
