# 🧠 Modular Anomaly Detection Framework (MNIST Benchmark)

## Overview

This repository implements a **modular, end-to-end anomaly detection framework** using deep autoencoders, with a focus on **reconstruction-based anomaly detection and localization**.

The project is designed as a **research-ready framework**, not a single experiment.  
All components—datasets, corruptions, models, metrics, and visualizations—are implemented in a reusable and extensible manner.

The MNIST dataset is used as a **controlled benchmark** to study:
- What constitutes an anomaly
- How reconstruction error separates normal vs anomalous samples
- How architectural choices (e.g. bottleneck size) affect detection performance
- How anomalies can be **localized spatially** via error heatmaps

---

## Key Contributions

- ✅ End-to-end anomaly detection pipeline  
- ✅ Modular corruption (anomaly) generators  
- ✅ Autoencoder-based reconstruction model  
- ✅ ROC–AUC based evaluation  
- ✅ Pixel-level anomaly localization via error heatmaps  
- ✅ Robustness studies:
  - Bottleneck size vs performance  
  - Contamination robustness  
  - Difficulty (noise strength) scaling  
- ✅ Script-based reproducibility (no notebook dependency)

---
## Repository Structure
```text
.
├── src/
│   ├── models/
│   │   └── autoencoder.py        # Autoencoder architecture
│   ├── utils/
│   │   ├── corruptions.py        # Anomaly generation functions
│   │   ├── metrics.py            # Evaluation metrics (ROC, AUC)
│   │   └── heatmaps.py           # Error heatmap & localization utilities
│   └── datasets/
│       └── mnist.py              # MNIST loading utilities
│
├── experiments/
│   └── baseline_mnist.py         # End-to-end experiment script
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_anomaly_examples.ipynb
│   └── 03_autoencoder_baseline.ipynb
│
├── results/
│   ├── anomalies/                # Heatmaps & localization outputs
│   ├── plots/                    # ROC, robustness, ablation plots
│   └── dataset_stats.json
│
├── requirements.txt
└── README.md
```
---
## Methodology

### 1. Problem Definition

An anomaly is defined as an input that **deviates from the normal data distribution** and is therefore poorly reconstructed by a model trained only on normal data.

---

### 2. Model: Autoencoder

A convolutional autoencoder is trained to reconstruct MNIST digits.

- Encoder compresses the image into a low-dimensional bottleneck  
- Decoder reconstructs the image from this representation  
- Reconstruction error serves as the anomaly score  

---

### 3. Anomaly Generation

Anomalies are synthetically generated using controlled corruptions:

- **Pixel Dropout** – random pixel masking  
- **Gaussian Noise** – additive noise with adjustable variance  
- **Stripe Corruption** – full row/column corruption  
- **Random Patch Occlusion** – localized missing regions  

These corruptions allow controlled study of anomaly difficulty.

---

### 4. Detection & Evaluation

- Reconstruction error is computed per image  
- ROC curves are generated using normal vs anomalous samples  
- Area Under Curve (AUC) is used as the primary metric  

---

### 5. Localization

Pixel-wise reconstruction error maps are visualized as heatmaps to **localize anomalous regions** within images.

---

### 6. Robustness Experiments

The framework includes systematic studies of:

- **Bottleneck size vs AUC**  
- **Training contamination vs detection performance**  
- **Anomaly difficulty scaling**  

---

## Results Summary

- Near-perfect separation achieved for strong anomalies (AUC ≈ 1.0)  
- Performance degrades gracefully with increasing anomaly difficulty  
- Smaller bottlenecks encourage better anomaly sensitivity  
- Error heatmaps accurately localize corrupted regions  

---

## How to Run

### 1. Setup Environment

```bash
pip install -r requirements.txt
```
### 2. Run Full Experiment

```bash
python experiments/baseline_mnist.py
```
This will automatically:

- Train the autoencoder  
- Generate anomalies  
- Compute ROC & AUC  
- Save plots and heatmaps to `results/`  

---

## Design Philosophy

- **Modularity** over monolithic notebooks  
- **Reproducibility** over ad-hoc experimentation  
- **Clarity** over excessive abstraction  
- **Research-first**, framework-second  

---

## Future Extensions

This framework is intentionally designed to support:

- Additional datasets (medical imaging, scientific data)  
- Alternative models (VAEs, diffusion-based methods)  
- Semi-supervised and weakly-supervised anomaly detection  
- Benchmarking across domains  

---

## Author

**Ajay Bandiwaddar**  
(GSoC 2026 Applicant)

---

## License

This project is released for research and educational use.

---