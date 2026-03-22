# FASG-Seg: Frequency-Aware Feature Refinement with Contextual Feature Propagation for Unsupervised Semantic Segmentation


Official PyTorch implementation of **FASG-Seg**.


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

# Install the rest of the required dependencies
pip install -r requirements.txt
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



