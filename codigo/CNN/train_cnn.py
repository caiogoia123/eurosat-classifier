import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(42)

# ---------- Caminhos ----------
DATA_DIR = Path("dataset/raw/EuroSAT")   # ajuste se rodar de dentro de notebooks/
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device)

# ---------- Dataset ----------
class EuroSATDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["Filename"]).convert("RGB")
        return self.transform(img), row["Label"]

transform = transforms.ToTensor()
train_ds = EuroSATDataset(DATA_DIR / "train.csv", DATA_DIR, transform)
val_ds   = EuroSATDataset(DATA_DIR / "validation.csv", DATA_DIR, transform)
print(f"Treino: {len(train_ds)} | Validação: {len(val_ds)}")

# ---------- DataLoader ----------
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

# ---------- Modelo ----------
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
print("Parâmetros:", sum(p.numel() for p in model.parameters()))

# ---------- Loss e otimizador ----------
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float("inf")
count = 0
patience = 5

# ---------- Treino + validação ----------
for epoch in range(40):
    start = time.time()

    # ----- treino -----
    model.train()
    train_loss = 0
    train_correct = 0                                                   
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (preds.argmax(dim=1) == labels).sum().item()   

    # ----- validação -----
    model.eval()
    val_loss = 0                                                        
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)                                         # ← guardar as notas
            val_loss += loss_fn(preds, labels).item()                   
            val_correct += (preds.argmax(dim=1) == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    
    print(f"Época {epoch:2d} | "
        f"treino: loss {train_loss / len(train_loader):.3f}, acc {train_correct / len(train_ds):.2%} | "
        f"val: loss {val_loss:.3f}, acc {val_correct / len(val_ds):.2%} | "
        f"{time.time() - start:.0f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        count = 0
        torch.save(model.state_dict(), "models/best_model.pt")
    else:
        count += 1
    
    if count == patience:
        print(f"Early stopping: {patience} épocas sem melhora. Melhor val_loss = {best_val_loss:.3f}")
        break

# ---------- Carregar o melhor modelo ----------
model.load_state_dict(torch.load("models/best_model.pt"))

# ---------- Testar uma imagem ----------
label_map = json.loads((DATA_DIR / "label_map.json").read_text())
idx_to_class = {v: k for k, v in label_map.items()}

img, label = val_ds[0]
with torch.no_grad():
    probs = model(img.unsqueeze(0).to(device)).softmax(dim=1)[0]
idx = probs.argmax().item()
print(f"\nReal: {idx_to_class[label]} | Previsto: {idx_to_class[idx]} ({probs[idx]:.1%})")