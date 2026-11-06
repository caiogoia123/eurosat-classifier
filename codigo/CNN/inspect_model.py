import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

torch.set_printoptions(precision=3, sci_mode=False)

DATA_DIR = Path("dataset/raw/EuroSAT")
MODEL_PATH = Path("models/best_model.pt")
IMG_PATH = DATA_DIR / "AnnualCrop/AnnualCrop_142.jpg"

# ---------------------------------------------------------------
# 1. Modelo, pesos e label_map
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

with open(DATA_DIR / "label_map.json") as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

# ---------------------------------------------------------------
# 2. Uma imagem real
# ---------------------------------------------------------------
img = Image.open(IMG_PATH).convert("RGB")
x0 = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, 64, 64]

print("=" * 70)
print(f"ENTRADA: {IMG_PATH.name}  formato {tuple(x0.shape)}")
print("Pixels 3x3 do canto superior esquerdo (R, G, B):")
print(x0[0, :, :3, :3])

# ---------------------------------------------------------------
# 3. Passo a passo: camada por camada
# ---------------------------------------------------------------
saidas = []
x = x0
with torch.no_grad():
    for i, layer in enumerate(model):
        entrada = x
        x = layer(x)
        saidas.append(x)

        nome = layer.__class__.__name__
        n_params = sum(p.numel() for p in layer.parameters())
        print("\n" + "-" * 70)
        print(f"[{i:2d}] {nome:10} {tuple(entrada.shape)} -> {tuple(x.shape)} | parâmetros: {n_params}")
        print(f"     min {x.min():.3f} | max {x.max():.3f} | média {x.mean():.3f} | zeros {(x == 0).float().mean():.1%}")
        if x.dim() == 4:  # ainda é "imagem": [lote, canais, altura, largura]
            print("     canal 0, canto 4x4:")
            print(x[0, 0, :4, :4])
        else:             # já virou lista: [lote, números]
            print(f"     primeiros 8 valores: {x[0, :8]}")

# ---------------------------------------------------------------
# 4. Conferindo as contas "na mão"
# ---------------------------------------------------------------
conv1, linear1 = model[0], model[10]
print("\n" + "=" * 70)
print("CONFERÊNCIAS")

# 4a. Um número da Conv 1, feito na mão
lin, col = 10, 10
w = conv1.weight[0]   # filtro 0: [3, 3, 3] -> 27 pesos
b = conv1.bias[0]
patch = x0[0, :, lin - 1:lin + 2, col - 1:col + 2]  # os 3x3 pixels em volta, nos 3 canais
with torch.no_grad():
    por_canal = (patch * w).sum(dim=(1, 2))
    manual = por_canal.sum() + b
print(f"\n[Conv 1] filtro 0, posição ({lin},{col})")
print(f"  R: {por_canal[0]:.3f} | G: {por_canal[1]:.3f} | B: {por_canal[2]:.3f} | bias: {b:.3f}")
print(f"  na mão: {manual:.4f}  |  PyTorch: {saidas[0][0, 0, lin, col]:.4f}")

# 4b. ReLU: quantos negativos viraram zero
print(f"\n[ReLU 1] negativos antes: {(saidas[0] < 0).float().mean():.1%} | zeros depois: {(saidas[1] == 0).float().mean():.1%}")

# 4c. MaxPool: o maior do bloco 2x2
bloco = saidas[1][0, 0, :2, :2]
print(f"\n[MaxPool 1] bloco 2x2:\n{bloco}")
print(f"  maior: {bloco.max():.3f}  |  saída do pool: {saidas[2][0, 0, 0, 0]:.3f}")

# 4d. Flatten: mapa 5, linha 2, coluna 3 -> posição 339
print(f"\n[Flatten] mapa 5, linha 2, coluna 3: {saidas[8][0, 5, 2, 3]:.4f}  |  posição 339: {saidas[9][0, 339]:.4f}")

# 4e. O neurônio mais "aceso" da Linear(4096, 128), feito na mão
n = saidas[10][0].argmax().item()
with torch.no_grad():
    produtos = saidas[9][0] * linear1.weight[n]   # 4096 produtos
    manual = produtos.sum() + linear1.bias[n]
print(f"\n[Linear 1] neurônio {n}  na mão: {manual:.4f}  |  PyTorch: {saidas[10][0, n]:.4f}")
top = produtos.topk(5)
print("  as 5 entradas que mais empurraram para cima:")
for pos in top.indices.tolist():
    print(f"    posição {pos:4d} (mapa {pos // 64:2d}, linha {pos % 64 // 8}, coluna {pos % 8}) "
          f"valor {saidas[9][0, pos]:.3f} × peso {linear1.weight[n, pos]:+.3f} = {produtos[pos]:+.3f}")

# ---------------------------------------------------------------
# 5. Notas finais e o chute
# ---------------------------------------------------------------
notas = saidas[-1][0]
probs = torch.softmax(notas, dim=0)
print("\n" + "=" * 70)
print("NOTAS FINAIS (da maior para a menor)")
for idx in notas.argsort(descending=True).tolist():
    print(f"  {idx}  {idx_to_class[idx]:22} nota {notas[idx]:6.2f} | prob {probs[idx]:6.1%}")
chute = notas.argmax().item()
print(f"\nChute: {chute} = {idx_to_class[chute]}  (imagem real: AnnualCrop)")
print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters())}")