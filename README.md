## Introduction

This repository contains the implementation of the paper "An Adaptive Canny Edge Detection Architecture with Dual Attention for Precise Segmentation of Individual Aquaculture Ponds". The proposed UNet-Edge model integrates adaptive Canny edge detection with a dual attention mechanism to achieve precise segmentation of individual aquaculture ponds. 


## Directory Structure

```
.
├── data/                     # Dataset directory
├── work/                     # Working directory
│   ├── config.py            # Configuration file with hyperparameters and paths
│   ├── evaluator.py         # Model evaluation module
│   ├── losses.py            # Loss function definitions
│   ├── main.py              # Main entry point
│   ├── metrics.py           # Evaluation metrics implementation
│   ├── predict.py           # Prediction script
│   ├── trainer.py           # Training module
│   ├── model/               # Model definitions
│   │   ├── __init__.py
│   │   └── unet_edge.py     # UNet-Edge model implementation
│   └── utils/               # Utility modules
│       └── dataset.py       # Dataset loading and preprocessing
└── README.md
```

## Requirements

- Python 3.7+
- PaddlePaddle 2.4+
- NumPy
- OpenCV
- Pillow
- tqdm
- VisualDL
- scikit-learn

## Data Preparation

Download the dataset from the following link:
- Link: https://pan.baidu.com/s/1_olRwEgAh1rj685lKaInPg?pwd=1210
- Extract and place the data in the `data/` directory

Expected dataset structure:
```
data/
├── Experiment1/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

## Quick Start

### Training

Train the model with default settings:
```bash
python work/main.py
```

Key parameters:
- `-e, --epochs`: Number of training epochs 
- `-b, --batch-size`: Batch size 
- `-l, --learning-rate`: Learning rate 
- `-v, --validation`: Validation split percentage 
- `--mode`: Running mode (train/evaluate/test)



## Evaluation Metrics

The following metrics are computed:
- Dice Coefficient
- Intersection over Union (IoU)
- F1 Score
- Pixel Accuracy
- Overall Accuracy (OA)
- Matthews Correlation Coefficient (MCC)
- Precision and Recall


## Contact

For questions or suggestions, please contact: [1152348045@qq.com]
