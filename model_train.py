# train.py
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

# === Ayarlar ===
DATA_DIR = "face_dataset"
MODEL_PATH = "face_model.pt"
EPOCHS = 15
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Augmentation ve Normalizasyon ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset ve Split ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_val = int(len(dataset) * VALIDATION_SPLIT)
num_train = len(dataset) - num_val
train_data, val_data = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# === Model Hazırla ===
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(DEVICE)

# === Her katmanı eğit — full fine-tune ===
for param in model.parameters():
    param.requires_grad = True

# === Loss ve Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# === Eğitim Başlasın ===
print(f"🚀 Eğitim başlıyor (cihaz: {DEVICE})")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

    # === Validation ===
    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

        val_acc = correct_val / total_val * 100
        print(f"           👉 Validation Accuracy: {val_acc:.2f}%")

# === Modeli Kaydet ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Eğitim tamamlandı. Model kaydedildi: {MODEL_PATH}")
