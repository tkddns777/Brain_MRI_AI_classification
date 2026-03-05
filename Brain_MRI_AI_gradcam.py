import torch
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json

# =====================================================
# Config
# =====================================================

MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\model\resnet_seed20_best_acc0.9902.pth"

TEST_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\Test")

SAVE_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\gradcam_results")

JSON_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Brain_MRI\gradcam_json")

CLASS_NAMES = ['glioma','meningioma','notumor','pituitary']

IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UNCERTAIN_THRESHOLD = 0.15

SAVE_DIR.mkdir(parents=True,exist_ok=True)
JSON_DIR.mkdir(parents=True,exist_ok=True)

print("DEVICE:",DEVICE)

# =====================================================
# Border Crop
# =====================================================

class BorderCrop:

    def __init__(self,ratio=0.07):
        self.ratio=ratio

    def __call__(self,img):

        img=np.array(img)

        h,w=img.shape[:2]

        dh=int(h*self.ratio)
        dw=int(w*self.ratio)

        cropped=img[dh:h-dh, dw:w-dw]

        return Image.fromarray(cropped)

# =====================================================
# Unicode safe image save
# =====================================================

def imwrite_unicode(path,img):

    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    ext=path.suffix if path.suffix else ".png"

    img=np.ascontiguousarray(img)

    if img.dtype!=np.uint8:
        img=np.clip(img,0,255).astype(np.uint8)

    ok,buf=cv2.imencode(ext,img)

    if not ok:
        raise RuntimeError("cv2.imencode failed")

    buf.tofile(str(path))

# =====================================================
# Model
# =====================================================

def get_model():

    model=models.resnet18()

    in_features=model.fc.in_features
    model.fc=torch.nn.Linear(in_features,len(CLASS_NAMES))

    target_layer=model.layer4[-1]

    return model,target_layer

model,target_layer=get_model()

state_dict=torch.load(MODEL_PATH,map_location=DEVICE)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded")

# =====================================================
# GradCAM Hook
# =====================================================

gradients=None
activations=None

def backward_hook(module,grad_input,grad_output):
    global gradients
    gradients=grad_output[0]

def forward_hook(module,input,output):
    global activations
    activations=output

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =====================================================
# Preprocessing (훈련과 동일)
# =====================================================

transform=transforms.Compose([

    BorderCrop(0.07),

    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])

])

# =====================================================
# Heatmap 분석
# =====================================================

def analyze_heatmap(cam):

    threshold = np.percentile(cam, 90)  #top 15% 활성화 영역 선택
    mask = cam > threshold

    ys,xs=np.where(mask)

    if len(xs)==0:

        return {
            "center_x":0,
            "center_y":0,
            "area_ratio":0,
            "bbox":[0,0,0,0]
        }

    center_x=float(xs.mean()/cam.shape[1])
    center_y=float(ys.mean()/cam.shape[0])

    area_ratio=float(mask.sum()/mask.size)

    x_min,x_max=xs.min(),xs.max()
    y_min,y_max=ys.min(),ys.max()

    bbox=[int(x_min),int(y_min),int(x_max),int(y_max)]

    return{
        "center_x":center_x,
        "center_y":center_y,
        "area_ratio":area_ratio,
        "bbox":bbox
    }

# =====================================================
# Brain region 추정
# =====================================================

def get_brain_region(cx,cy):

    if cx<0.33:
        x="Left"
    elif cx<0.66:
        x="Center"
    else:
        x="Right"

    if cy<0.33:
        y="Frontal"
    elif cy<0.66:
        y="Central"
    else:
        y="Posterior"

    return f"{x}-{y}"

# =====================================================
# GradCAM 생성
# =====================================================

def generate_gradcam(img_path,label):

    img=Image.open(img_path).convert("RGB")

    orig=np.array(img)

    x=transform(img).unsqueeze(0).to(DEVICE)

    output=model(x)

    probs=torch.softmax(output,dim=1)

    pred=torch.argmax(probs,dim=1).item()

    confidence=probs[0,pred].item()

    score=output[:,pred]

    model.zero_grad()

    score.backward()

    grads = gradients[0]
    acts = activations[0]

    # GradCAM++ 계산
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    sum_acts = torch.sum(acts, dim=(1,2), keepdim=True)

    eps = 1e-8

    alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)

    weights = torch.sum(alpha * torch.relu(grads), dim=(1,2))

    cam = torch.sum(weights[:,None,None] * acts, dim=0)

    cam = torch.relu(cam)

    cam=cam.detach().cpu().numpy()

    orig_h, orig_w = orig.shape[:2]
    cam = cv2.resize(cam,(orig_w, orig_h))

    cam = cam ** 4 # 활성화 영역 강조

    cam-=cam.min()
    cam/=(cam.max()+1e-8)

    hotspot=analyze_heatmap(cam)

    cx=hotspot["center_x"]
    cy=hotspot["center_y"]
    area_ratio=hotspot["area_ratio"]
    bbox=hotspot["bbox"]

    region=get_brain_region(cx,cy)

    heatmap=np.uint8(cam*255)
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    # overlay용 동일 preprocessing
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    overlay=cv2.addWeighted(orig_bgr,0.6,heatmap,0.4,0)

    if bbox!=[0,0,0,0]:

        cv2.rectangle(
            overlay,
            (bbox[0],bbox[1]),
            (bbox[2],bbox[3]),
            (0,255,0),
            2
        )

    filename=Path(img_path).name

    gt=CLASS_NAMES[label]
    pred_name=CLASS_NAMES[pred]

    text=f"GT:{gt} Pred:{pred_name} ({confidence:.3f})"

    cv2.putText(
        overlay,
        text,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    probs_np=probs[0].detach().cpu().numpy()

    prob_dict={
        CLASS_NAMES[i]:float(probs_np[i])
        for i in range(len(CLASS_NAMES))
    }

    sorted_probs=sorted(prob_dict.items(),key=lambda x:x[1],reverse=True)

    top1=sorted_probs[0]
    top2=sorted_probs[1]

    diff=top1[1]-top2[1]

    uncertain=diff<UNCERTAIN_THRESHOLD

    result={

        "image":filename,
        "ground_truth":gt,
        "prediction":pred_name,
        "confidence":float(confidence),
        "class_probabilities":prob_dict,
        "top_candidates":[top1,top2],
        "uncertain":uncertain,
        "activation_center_x":cx,
        "activation_center_y":cy,
        "activation_area_ratio":area_ratio,
        "activation_region":region,
        "bbox":bbox
    }

    json_path=JSON_DIR/f"{Path(filename).stem}.json"

    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=4,ensure_ascii=False)

    # 클래스 기준 저장
    gt_class = CLASS_NAMES[label]

    out_dir = SAVE_DIR / gt_class

    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / filename

    imwrite_unicode(save_path,overlay)

# =====================================================
# Run
# =====================================================

print("\nStarting GradCAM generation\n")

dataset=datasets.ImageFolder(TEST_DIR)

for img_path,label in tqdm(dataset.samples,total=len(dataset.samples)):

    generate_gradcam(img_path,label)

print("\nGradCAM generation complete")