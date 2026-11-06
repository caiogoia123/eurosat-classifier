import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

DATA_DIR = Path("dataset/raw/EuroSAT")
MODEL_PATH = Path("models/best_model.pt")
IMG_PATH = DATA_DIR / "AnnualCrop/AnnualCrop_142.jpg"   # troque para testar outras imagens

# ---------------------------------------------------------------
# 1. Modelo treinado
# ---------------------------------------------------------------
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---------------------------------------------------------------
# 2. Passa a imagem e guarda a saída de cada camada
# ---------------------------------------------------------------
img = Image.open(IMG_PATH).convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0)   # [1, 3, 64, 64]

saidas = []
with torch.no_grad():
    for layer in model:
        x = layer(x)
        saidas.append(x)

# saídas depois de cada ReLU (o que "acendeu" em cada bloco)
blocos = {
    "Bloco 1 (depois do ReLU) - 16 mapas de 64x64": saidas[1][0],
    "Bloco 2 (depois do ReLU) - 32 mapas de 32x32": saidas[4][0],
    "Bloco 3 (depois do ReLU) - 64 mapas de 16x16": saidas[7][0],
}


# ---------------------------------------------------------------
# 3. Função que desenha vários mapas em grade
# ---------------------------------------------------------------
def mostrar_grade(imagens, titulo, colunas=8, cmap="viridis"):
    n = len(imagens)
    linhas = math.ceil(n / colunas)
    fig, axes = plt.subplots(linhas, colunas, figsize=(colunas * 1.6, linhas * 1.6 + 0.6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(imagens[i], cmap=cmap)
            ax.set_title(str(i), fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


# 3a. A imagem original
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(img)
ax.set_title(IMG_PATH.name)
ax.axis("off")

# 3b. Os 16 filtros da Conv 1 vistos como "mini imagens" RGB 3x3
w = model[0].weight.detach()                     # [16, 3, 3, 3]
filtros = []
for f in w:
    f = (f - f.min()) / (f.max() - f.min())      # escala para 0-1, só para conseguir desenhar
    filtros.append(f.permute(1, 2, 0))           # [3, 3, 3] -> [altura, largura, canais]
mostrar_grade(filtros, "Os 16 filtros da Conv 1 (pesos em cores)", cmap=None)

# 3c. Os mapas de cada bloco
for titulo, mapas in blocos.items():
    mostrar_grade(mapas, titulo)

# 3d. Lado a lado: imagem x mapas mais "acesos" do bloco 1
mapas = saidas[1][0]
top = mapas.mean(dim=(1, 2)).topk(4).indices.tolist()
fig, axes = plt.subplots(1, 5, figsize=(13, 3))
axes[0].imshow(img)
axes[0].set_title("original")
for ax, c in zip(axes[1:], top):
    ax.imshow(mapas[c], cmap="viridis")
    ax.set_title(f"mapa {c} (média {mapas[c].mean():.2f})")
for ax in axes:
    ax.axis("off")
fig.tight_layout()

plt.show()