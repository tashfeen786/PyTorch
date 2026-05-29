# 🔥 PyTorch — Complete Deep Learning Repository

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![GPU](https://img.shields.io/badge/GPU-CUDA%20Enabled-green?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

> 🧠 A complete hands-on PyTorch deep learning repository covering
> everything from tensor operations and autograd to CNNs, RNNs, LSTMs,
> Transfer Learning, GPU training, and hyperparameter optimization
> with Optuna — built through real projects on Fashion MNIST dataset.

---

## 📚 Complete Topics Index

| # | Notebook | Topic | Level |
|---|----------|-------|-------|
| 1 | `1_Tensors_in_Pytorch.ipynb` | Tensors — creation, ops, GPU | 🟢 Beginner |
| 2 | `pytorch_autograd.ipynb` | Autograd & Backpropagation | 🟢 Beginner |
| 3 | `dataset_and_dataloader_demo.ipynb` | Custom Dataset & DataLoader | 🟡 Intermediate |
| 4 | `pytorch_training_pipeline.ipynb` | Basic Training Pipeline | 🟡 Intermediate |
| 5 | `pytorch_training_pipeline_using_dataset_...ipynb` | Training Pipeline + DataLoader | 🟡 Intermediate |
| 6 | `pytorch_training_pipeline_using_nn_mod...ipynb` | Training Pipeline + nn.Module | 🟡 Intermediate |
| 7 | `ann_fashion_mnist_pytorch_gpu.ipynb` | ANN on Fashion MNIST — GPU | 🟠 Advanced |
| 8 | `ann_fashion_mnist_pytorch_gpu_optimiz...ipynb` | ANN + BatchNorm + Dropout | 🟠 Advanced |
| 9 | `ann_fashion_mnist_pytorch_gpu_optimiz...ipynb` | ANN + Optuna Hyperparameter Tuning | 🔴 Expert |
| 10 | `cnn_fashion_mnist_pytorch_gpu.ipynb` | CNN on Fashion MNIST — GPU | 🟠 Advanced |
| 11 | `cnn_optuna.ipynb` | CNN + Optuna Tuning | 🔴 Expert |
| 12 | `pytorch_lstm_next_word_predictor.ipynb` | LSTM — Next Word Prediction | 🔴 Expert |
| 13 | `pytorch_rnn_based_qa_system.ipynb` | RNN — Question Answering System | 🔴 Expert |
| 14 | `transfer_learning_fashion_mnist_pytorch...ipynb` | Transfer Learning — ResNet18 | 🔴 Expert |

### 📊 Datasets
| File | Description |
|------|-------------|
| `fmnist_small.csv` | Fashion MNIST dataset (13MB) |
| `100_Unique_QA_Dataset.csv` | Custom QA dataset for RNN system |

---

## 🔍 Topic Deep Dives

### 1. 🔢 Tensors
```python
import torch

# Create tensors
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
t = t.to(device)

# Operations
print(t.shape)       # torch.Size([2, 2])
print(t @ t)         # Matrix multiplication
print(t.mean())      # Mean
print(t.reshape(-1)) # Flatten
```

---

### 2. ⚡ Autograd
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

y.backward()       # Backpropagation
print(x.grad)      # dy/dx = 2x+2 = 8.0

# Always zero grad before next step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

### 3. 📦 Custom Dataset & DataLoader
```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class FashionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        pixels = self.data.iloc[idx, 1:].values / 255.0
        return torch.tensor(pixels, dtype=torch.float32), label

dataset = FashionDataset("fmnist_small.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### 4. 🏗️ Training Pipeline — nn.Module
```python
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Training loop
model = NeuralNet(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
```

---

### 5. 🖼️ CNN on Fashion MNIST (GPU)
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc_block(self.conv_block(x))
```

---

### 6. 🎯 Optuna — Hyperparameter Tuning
```python
import optuna

def objective(trial):
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden     = trial.suggest_categorical("hidden", [128, 256, 512])
    dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = NeuralNet(784, hidden, 10).to(device)
    # ... training loop with above params ...
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
print("Best accuracy:", study.best_value)
```

---

### 7. 🔄 LSTM — Next Word Predictor
```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden
```

---

### 8. 🤖 RNN — QA System
```python
# Uses 100_Unique_QA_Dataset.csv
class RNNQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```

---

### 9. 🚀 Transfer Learning — ResNet18
```python
import torchvision.models as models

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 10 classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

model = model.to(device)
# Only final layer trains — fast & efficient
```

---

## 📊 Results Summary

| Model | Dataset | Accuracy | Key Technique |
|-------|---------|----------|---------------|
| ANN Basic | Fashion MNIST | ~87% | GPU Training |
| ANN Optimized | Fashion MNIST | ~88% | BatchNorm + Dropout |
| ANN + Optuna | Fashion MNIST | ~89% | Hyperparameter Tuning |
| CNN GPU | Fashion MNIST | ~91% | Conv2d + MaxPool |
| CNN + Optuna | Fashion MNIST | ~92%+ | Full HP Optimization |
| Transfer Learning | Fashion MNIST | ~93%+ | ResNet18 Pretrained |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.x |
| GPU Training | CUDA |
| HP Tuning | Optuna |
| Datasets | Fashion MNIST · Custom QA CSV |
| Models | ANN · CNN · RNN · LSTM |
| Transfer Learning | torchvision ResNet18 |
| Notebooks | Jupyter / Google Colab |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/tashfeen786/PyTorch.git
cd PyTorch

# Install dependencies
pip install torch torchvision jupyter optuna pandas numpy matplotlib

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Launch notebooks
jupyter notebook
```

---

## 🏗️ Project Structure

```
PyTorch/
│
├── 📊 Foundations
│   ├── 1_Tensors_in_Pytorch.ipynb
│   └── pytorch_autograd.ipynb
│
├── 📦 Data Handling
│   ├── dataset_and_dataloader_demo.ipynb
│   ├── fmnist_small.csv                     # Fashion MNIST (13MB)
│   └── 100_Unique_QA_Dataset.csv            # Custom QA dataset
│
├── 🏗️ Training Pipelines
│   ├── pytorch_training_pipeline.ipynb
│   ├── pytorch_training_pipeline_using_dataset_...ipynb
│   └── pytorch_training_pipeline_using_nn_mod...ipynb
│
├── 🧠 ANN Projects
│   ├── ann_fashion_mnist_pytorch_gpu.ipynb
│   ├── ann_fashion_mnist_pytorch_gpu_optimiz...ipynb
│   └── ann_fashion_mnist_pytorch_gpu_optimiz...ipynb  ← Optuna
│
├── 🖼️ CNN Projects
│   ├── cnn_fashion_mnist_pytorch_gpu.ipynb
│   └── cnn_optuna.ipynb
│
├── 🔄 Sequential Models
│   ├── pytorch_lstm_next_word_predictor.ipynb
│   └── pytorch_rnn_based_qa_system.ipynb
│
└── 🚀 Transfer Learning
    └── transfer_learning_fashion_mnist_pytorch...ipynb
```

---

## 🔗 Real-World Connection

| Skill Learned | Applied In |
|--------------|-----------|
| CNN + GPU | [HelmetEye FYP](https://github.com/tashfeen786/HelmetEye) — YOLOv12 based |
| Custom DataLoader | ML pipeline projects |
| Transfer Learning | Production CV systems |
| LSTM / RNN | Sequence modeling & NLP |
| Optuna Tuning | Model optimization in production |

---

## 👨‍💻 Author

**Tashfeen Aziz** — AI/ML Engineer & Python Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/tashfeen-aziz-b51361292)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/tashfeen786)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:tashfeen247@gmail.com)

---

⭐ **If you found this helpful, please give it a star!**
