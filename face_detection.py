import cv2
import numpy as np
import pyautogui
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Modeli yükle
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(pil_img):
    img_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_tensor)
    return emb.squeeze().numpy()

# Referans embed'ler
emb_normal = get_embedding(Image.open("normal.jpg").convert("RGB"))
emb_shocked = get_embedding(Image.open("shocked.jpg").convert("RGB"))

def detect_face():
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        return face_pil
    return None

def is_shocked():
    face_img = detect_face()
    if face_img:
        emb_test = get_embedding(face_img)
        dist_normal = np.linalg.norm(emb_test - emb_normal)
        dist_shocked = np.linalg.norm(emb_test - emb_shocked)
        return dist_shocked < dist_normal
    return False
