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
```

**2. Create a virtual environment**
```bash
conda create -n fasg python=3.8 -y
conda activate fasg
```
**3. Install PyTorch and dependencies**
```bash
# Install PyTorch (Please adjust the CUDA version according to your machine)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install OpenAI CLIP
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)

# Install other required packages
pip install numpy PyYAML Pillow opencv-python scikit-learn
```

**📂 Data Preparation**
Please download the PASCAL VOC 2012, PASCAL Context, or Cityscapes datasets and organize them as follows. Set the dataset paths in the corresponding configuration files located in the config/ directory.
```bash
datasets/
├── VOC2012/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/
├── context/
└── cityscapes/
```

**🚀 Usage**
Training
To train the FASG-Seg model on PASCAL VOC 2012, run the following command:
```bash
python train.py --cfg config/voc_train_ori_cfg.yaml
```
Evaluation
To evaluate the trained model, make sure you have specified the correct LOAD_PATH in your test configuration file, then run:
```bash
python test.py --cfg config/voc_test_ori_cfg.yaml --model FASGSeg
```



