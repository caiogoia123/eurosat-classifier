# EuroSAT Classifier

Projeto de estudo de **deep learning**: classificação de imagens de satélite em 10 tipos de uso do solo, usando uma CNN construída do zero em PyTorch.

## Dataset

[EuroSAT (Kaggle)](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset): 27.000 imagens RGB de 64×64 pixels do satélite Sentinel-2, divididas em 10 classes:

AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake

| Split | Imagens |
|---|---|
| Treino | 18.900 |
| Validação | 5.400 |
| Teste | 2.700 |

O dataset não está no repositório (cerca de 2 GB). Baixe pelo Kaggle e extraia em `dataset/raw/`.

## Estrutura

```
eurosat-classifier/
├── codigo/CNN/
│   ├── train_cnn.py            # treino com early stopping
│   ├── evaluate_test.py        # avaliação no conjunto de teste
│   ├── inspect_model.py        # passo a passo do modelo, camada por camada
│   └── show_features_model.py  # visualização dos filtros e mapas de ativação
├── dataset/raw/EuroSAT/        # imagens + CSVs (não versionado)
├── models/                     # modelo treinado (não versionado)
└── requirements.txt
```

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Todos os scripts rodam a partir da raiz do projeto:

```bash
python codigo/CNN/train_cnn.py       # treina e salva models/best_model.pt
python codigo/CNN/evaluate_test.py   # avalia no teste
```

## Modelo

CNN simples com 3 blocos convolucionais e 549.290 parâmetros:

```
[3, 64, 64] → Conv+ReLU+Pool ×3 → [64, 8, 8] → Flatten → Linear(128) → Linear(10)
```

- Loss: CrossEntropyLoss
- Otimizador: Adam (lr = 0,001)
- Batch: 64
- Early stopping: paciência de 5 épocas sobre a loss de validação
- Treinado em CPU

## Resultados

| Conjunto | Acurácia | Loss |
|---|---|---|
| Validação (melhor época: 14) | 86,39% | 0,409 |
| **Teste** | **86,59%** | **0,398** |

## Próximos passos

- [ ] Matriz de confusão e acurácia por classe
- [ ] Curvas de treino (loss e acurácia por época)
- [ ] Data augmentation e Dropout
- [ ] BatchNorm e scheduler de learning rate
- [ ] Transfer learning (ResNet18)
- [ ] Versão com as 13 bandas espectrais