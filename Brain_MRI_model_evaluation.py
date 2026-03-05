import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)

from pathlib import Path

# =====================================================
# 설정 (여기 수정)
# =====================================================

MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\model\resnet_seed0_best_acc0.9741.pth"

TEST_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\Test")

SAVE_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\evaluation_results")

CLASS_NAMES = ['glioma','meningioma','notumor','pituitary']

IMG_SIZE = 224
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 모델 로드
# =====================================================

def get_model():

    model = models.resnet50()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))

    return model


model = get_model()

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded")

# =====================================================
# 데이터 로드
# =====================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(dataset))

# =====================================================
# 추론
# =====================================================

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# =====================================================
# Accuracy
# =====================================================

accuracy = accuracy_score(all_labels, all_preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="macro"
)

auc = roc_auc_score(
    all_labels,
    all_probs,
    multi_class="ovr"
)

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(all_labels, all_preds)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(SAVE_DIR / "confusion_matrix.csv")

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# =====================================================
# Classification Report
# =====================================================

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
)

with open(SAVE_DIR / "classification_report.txt","w") as f:
    f.write(report)

# =====================================================
# ROC Curve
# =====================================================

plt.figure(figsize=(8,6))

for i in range(len(CLASS_NAMES)):

    fpr, tpr, _ = roc_curve(
        (all_labels == i).astype(int),
        all_probs[:, i]
    )

    plt.plot(fpr, tpr, label=CLASS_NAMES[i])

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig(SAVE_DIR / "roc_curve.png", dpi=300)
plt.close()

# =====================================================
# 예측 결과 저장
# =====================================================

pred_df = pd.DataFrame({
    "True Label": all_labels,
    "Predicted Label": all_preds
})

pred_df.to_csv(SAVE_DIR / "predictions.csv", index=False)

# =====================================================
# Metric 저장
# =====================================================

with open(SAVE_DIR / "metrics.txt","w") as f:

    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"ROC AUC: {auc}\n")

print("\nEvaluation results saved to:")
print(SAVE_DIR)