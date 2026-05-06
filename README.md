# 🔥 PyTorch — Deep Learning Fundamentals

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Fundamentals-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

> 🧠 Hands-on PyTorch notebooks covering the **core building blocks
> of deep learning** — from tensor operations to automatic
> differentiation with Autograd — the foundation of every
> neural network training loop.

---

## 📚 What's Covered

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|-------------|
| 1 | `1_Tensors_in_Pytorch.ipynb` | Tensors | Creation, operations, GPU support |
| 2 | `pytorch_autograd.ipynb` | Autograd | Gradients, backpropagation, `requires_grad` |

---

## 🧩 Topic Deep Dive

### 1. 🔢 Tensors in PyTorch
The **fundamental data structure** of all PyTorch operations —
multi-dimensional arrays that run on CPU or GPU.

```python
import torch

# Creating tensors
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros = torch.zeros(3, 3)
rand = torch.rand(2, 4)

# Operations
print(t.shape)        # torch.Size([2, 2])
print(t.dtype)        # torch.float32
print(t.device)       # cpu

# GPU support
if torch.cuda.is_available():
    t = t.to("cuda")

# Tensor math
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)          # tensor([5., 7., 9.])
print(torch.dot(a,b)) # tensor(32.) — dot product
print(a @ b)          # same as dot product

# NumPy bridge
import numpy as np
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)  # NumPy → Tensor
```

**Key Concepts:**
- Tensor shapes, dtypes, devices
- CPU vs GPU tensors (`.to("cuda")`)
- Tensor operations — add, multiply, matmul
- NumPy ↔ PyTorch bridge
- Broadcasting rules

---

### 2. ⚡ Autograd — Automatic Differentiation
PyTorch's **automatic gradient computation** engine —
the magic behind neural network training.

```python
import torch

# requires_grad=True → track operations for gradient
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# Compute gradients (backpropagation)
y.backward()

# dy/dx = 2x + 2 = 2(3) + 2 = 8
print(x.grad)   # tensor(8.)

# Gradient in neural network context
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass
y_pred = w * x + b
loss = (y_pred - 5.0) ** 2

# Backward pass — compute gradients
loss.backward()

print(f"dL/dw = {w.grad}")  # gradient wrt weight
print(f"dL/db = {b.grad}")  # gradient wrt bias
```

**Key Concepts:**
- `requires_grad=True` — tracking computations
- Computational graph — forward pass
- `.backward()` — backpropagation
- `.grad` — accessing gradients
- `torch.no_grad()` — disable gradient tracking (inference)
- Gradient accumulation and zeroing

---

## 🏗️ Project Structure

```
PyTorch/
│
├── 1_Tensors_in_Pytorch.ipynb    # Tensor ops, shapes, GPU
├── pytorch_autograd.ipynb        # Autograd, backprop, gradients
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/tashfeen786/PyTorch.git
cd PyTorch

# Install PyTorch (CPU)
pip install torch torchvision jupyter

# Install PyTorch (GPU - CUDA 11.8)
pip install torch torchvision --index-url \
  https://download.pytorch.org/whl/cu118

# Launch notebooks
jupyter notebook
```

---

## 🔗 Why PyTorch Matters

PyTorch is used in:
- ✅ **Computer Vision** — YOLO, ResNet, CNNs (used in HelmetEye FYP)
- ✅ **NLP & LLMs** — GPT, BERT, LLaMA training
- ✅ **Research** — most AI papers use PyTorch
- ✅ **Production** — Meta, Tesla, OpenAI all use PyTorch

---

## 🔮 Coming Soon

- [ ] Neural Network from scratch (`nn.Module`)
- [ ] Training loop — forward, loss, backward, optimize
- [ ] CNN for image classification
- [ ] Transfer learning with pretrained models
- [ ] YOLO integration with PyTorch

---

## 👨‍💻 Author

**Tashfeen Aziz** — AI/ML Engineer & Python Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/tashfeen-aziz-b51361292)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/tashfeen786)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:tashfeen247@gmail.com)

---

⭐ **If you found this helpful, please give it a star!**
