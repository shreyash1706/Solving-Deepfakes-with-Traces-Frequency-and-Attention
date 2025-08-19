# Solving Deepfakes with Traces, Frequency, and Attention

This repository contains **PyTorch implementations** of various models explored during my internship project on deepfake detection.  
The focus is on **combining trace extraction, frequency domain analysis, and attention mechanisms** to build effective detectors.  

👉 For a detailed walkthrough of the methodology, experiments, and insights, check out my blog post: [*Solving Deepfakes with Traces, Frequency, and Attention!*](https://medium.com/@Shreyash-Pawar/solving-deepfakes-with-traces-frequency-and-attention-e77f2a92f09a)

---

## 🚀 Overview
Deepfakes are a growing concern, and this repo includes **baseline models** (like ResNet, DenseNet) and **custom hybrids** (like AMTEN-Freq-CBAM) tested on datasets such as:

- **Hybrid Fake Face Dataset**  
- **Kaggle’s 140k Real/Fake Faces**  

The **hybrid model** achieved **~98.97% test accuracy** by integrating:
- **AMTENet** → Trace extraction  
- **VANet** → Spatial-frequency fusion  
- **CBAM** → Attention mechanism  

---

## 📂 Files and Models

- `AMTENNet.py` → Implementation of **Adaptive Manipulative Trace Extraction Network (AMTENet)** for low-level trace detection.  
- `AMTEN_freq_cbam.py` → The **hybrid model** combining AMTENet, frequency domain, and CBAM attention (best performer).  
- `AutoEncoder.py` → One-class autoencoder experiment (trained on real images for anomaly detection).  
- `DenseNet121.py` → DenseNet-121 variant for feature-dense classification.  
- `EffNetCBAM.py` → EfficientNet with CBAM attention integration.  
- `MCNet.py` → Manipulation Classification Network focusing on multi-domain features.  
- `Resnet50.py` → ResNet-50 baseline, with variations like **ELA preprocessing**.  
- `SFFN.py` → Shallow-FakeFaceNet for facial manipulation detection.  
- `VANet.py` → Visual Artifact Network for spatial and frequency domain learning.  

---

# ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/shreyash1706/Solving-Deepfakes-with-Traces-Frequency-and-Attention.git
cd Solving-Deepfakes-with-Traces-Frequency-and-Attention
```

Install dependencies (**Python 3.8+ recommended**):
```bash
pip install torch torchvision numpy matplotlib
```

> Note: Some models may require additional libraries like **scipy** for frequency transforms — install as needed.

---

# 🖥️ Usage

Each file is a **self-contained PyTorch module**.

Example: Loading and using the **hybrid model**:
```python
import torch
from AMTEN_freq_cbam import AMTENFC  

model = AMTENFC(num_classes=1)  
model.eval()  # For inference

output = model(rgb_input, dct_input)
```

➡️ Train or test using your datasets. Refer to the blog for preprocessing details like **DCT computation**.

---

# 📊 Results Snapshot

- **AMTEN-Freq-CBAM** → Achieved **98.97% accuracy**  
- Includes comparisons & **t-SNE plots** (see blog for details)  

---

# 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.



