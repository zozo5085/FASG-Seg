# FASG-Seg: Frequency-Aware Feature Refinement with Contextual Feature Propagation for Unsupervised Semantic Segmentation

> **News:** This paper has been submitted to the **IEEE International Conference on Systems, Man, and Cybernetics (IEEE SMC)**.

Official PyTorch implementation of **FASG-Seg**.

## 📝 Introduction

Unsupervised semantic segmentation (USS) has recently benefited from the strong vision-language alignment of CLIP. However, dense predictions often suffer from *boundary ambiguity* and *interior fragmentation*. 

**FASG-Seg** is a lightweight, pure-visual spatial refinement framework built upon a semantic-debiasing baseline. It tackles these issues without relying on heavy multi-stage distillation or additional self-supervised visual backbones.

### ✨ Main Components:
- **MSFA (Multi-Spectrum Feature Aggregation):** Injects high-frequency structural priors to sharpen object boundaries.
- **CFP (Contextual Feature Propagation):** Propagates reliable contextual responses to improve region-level completeness.
- **DFGP (Dynamic Frequency-Guided Pooling):** Stabilizes pseudo-label generation during training via boundary-aware scaling.
- **SGBR (Structure-Guided Boundary Refinement):** Suppresses semantic leakage at inference time combined with a Probability Drop (PD) strategy.

---

## ⚙️ Installation

The code has been tested with Python 3.8 and PyTorch 1.12+.

**1. Clone the repository**
```bash
git clone [https://github.com/zozo5085/FASG-Seg.git](https://github.com/zozo5085/FASG-Seg.git)
cd FASG-Seg


**2. Create a virtual environment**
conda create -n fasg python=3.8 -y
conda activate fasg

**3. Install PyTorch and dependencies**
