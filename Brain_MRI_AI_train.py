import os
import time
import copy
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# ==========================================================
# 설정 (여기만 수정)
# ==========================================================

TRAIN_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\Train")
TEST_DIR  = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\Test")
SAVE_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\model")

MODEL_NAME = "resnet"  # efficientnet / resnet / inception

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

IMG_SIZE = 224
VAL_RATIO = 0.3

SEEDS = [0,10,20,30,40]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)
print("GPU NAME:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ==========================================================
# border crop
class BorderCrop:

    def __init__(self, ratio=0.05):
        """
        ratio: 잘라낼 테두리 비율 (예: 0.05 = 5%)
        """
        self.ratio = ratio

    def __call__(self, img):

        img = np.array(img)

        h, w = img.shape[:2]

        dh = int(h * self.ratio)
        dw = int(w * self.ratio)

        cropped = img[dh:h-dh, dw:w-dw]

        return Image.fromarray(cropped)

# ==========================================================


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================

def get_model(name, num_classes):

    if name == "efficientnet":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "resnet":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "inception":

        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unknown model")

    return model


# ==========================================================

def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total

    return total_loss / len(loader), acc


# ==========================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = correct / total

    return total_loss / len(loader), acc, all_preds, all_labels, all_probs


# ==========================================================

def run_experiment(seed):

    print("\n=================================================")
    print(f"🚀 Running with SEED = {seed}")
    print("=================================================")

    set_seed(seed)

    writer = SummaryWriter(f"runs/mri_seed_{seed}")

    train_transform = transforms.Compose([
        BorderCrop(0.07),

        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        BorderCrop(0.07),

        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    class_names = full_train_dataset.classes
    num_classes = len(class_names)

    generator = torch.Generator().manual_seed(seed)

    val_size = int(len(full_train_dataset) * VAL_RATIO)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(MODEL_NAME, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    patience = 7
    patience_counter = 0

    for epoch in range(EPOCHS):

        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        elapsed = time.time() - start

        print(
            f"[Seed {seed}] Epoch {epoch+1}/{EPOCHS} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_acc:

            best_acc = val_acc
            patience_counter = 0

            save_path = SAVE_DIR / f"{MODEL_NAME}_seed{seed}_best_acc{best_acc:.4f}.pth"

            torch.save(model.state_dict(), save_path)

            print(f"Model saved to: {save_path}")

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print("Early stopping")
                break

    print("\nEvaluating TEST set")

    _, test_acc, preds, labels, probs = evaluate(model, test_loader, criterion)

    print(f"[Seed {seed}] Test accuracy: {test_acc:.4f}")

    writer.close()

    return test_acc


# ==========================================================

def main():

    results = []

    for seed in SEEDS:

        acc = run_experiment(seed)

        results.append(acc)

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")

    for s,a in zip(SEEDS, results):

        print(f"Seed {s} : {a:.4f}")

    print("\nMean accuracy :", np.mean(results))
    print("Std :", np.std(results))


if __name__ == "__main__":
    main()