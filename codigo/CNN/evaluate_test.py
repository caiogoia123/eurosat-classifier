from pathlib import Path
import json
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------- Caminhos ----------
DATA_DIR = Path("dataset/raw/EuroSAT")   # ajuste se rodar de dentro de notebooks/
DATA_MODEL = Path("models/best_model.pt")
label_map = json.loads((DATA_DIR / "label_map.json").read_text())
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
test_ds   = EuroSATDataset(DATA_DIR / "test.csv", DATA_DIR, transform)

print(f"Teste: {len(test_ds)}")


# ---------- DataLoader ----------
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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

print(model.load_state_dict(torch.load(DATA_MODEL)))

# ----- Teste -----

model.eval()
test_loss = 0                                                        
test_correct = 0
loss_fn = nn.CrossEntropyLoss()
prev_list = []
res_list = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)                              
        test_loss += loss_fn(preds, labels).item()                   
        test_correct += (preds.argmax(dim=1) == labels).sum().item()
        print(labels)           

test_loss = test_loss / len(test_loader)

print(f"Teste: loss {test_loss:.3f}, acc {test_correct / len(test_ds):.2%}")
print(label_map)
idx_label_map = {v: k for k, v in label_map.items()}
print(idx_label_map)
'''

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

'''