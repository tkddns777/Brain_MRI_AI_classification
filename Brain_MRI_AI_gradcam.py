import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# =====================================================
# 설정
# =====================================================

MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\model\resnet_seed0_best_acc0.9741.pth"

TEST_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\Test")

SAVE_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\gradcam_results")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Unicode-safe 저장 (OneDrive / 한글 경로 문제 해결)
# =====================================================

def imwrite_unicode(path, img):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix if path.suffix else ".png"

    img = np.ascontiguousarray(img)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img)

    if not ok:
        raise RuntimeError("cv2.imencode failed")

    buf.tofile(str(path))

    return str(path)


# =====================================================
# 모델 로드
# =====================================================

def get_model():

    model = models.resnet50()

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(CLASS_NAMES))

    target_layer = model.layer4[-1]

    return model, target_layer


model, target_layer = get_model()

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# =====================================================
# Hook 설정
# =====================================================

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =====================================================
# 이미지 전처리
# =====================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =====================================================
# Grad-CAM 생성
# =====================================================

def generate_gradcam(image_path, save_dir):

    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print("Image load failed:", image_path)
        return

    original = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)

    pred_class = torch.argmax(output, dim=1)

    score = output[:, pred_class]

    model.zero_grad()

    score.backward()

    grads = gradients.detach().cpu().numpy()[0]
    acts = activations.detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # heatmap 생성
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)

    # overlay
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

    name = Path(image_path).stem
    save_path = save_dir / f"{name}_overlay.png"

    imwrite_unicode(save_path, overlay)


# =====================================================
# 실행
# =====================================================

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

print("\nStarting Grad-CAM generation\n")

for class_dir in TEST_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    print("Processing class:", class_name)

    save_class_dir = SAVE_DIR / class_name
    save_class_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXT]

    print("Found images:", len(image_paths))

    for img_path in tqdm(image_paths):

        generate_gradcam(img_path, save_class_dir)

print("\nGrad-CAM generation complete")