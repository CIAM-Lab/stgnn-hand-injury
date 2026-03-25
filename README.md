# STGNN‑HTR: Spatio-Temporal Graph Neural Network for Hand Trauma Risk Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19217481.svg)](https://doi.org/10.5281/zenodo.19217481)
Official implementation of the paper:  
**"STGNN‑HTR: An Open-Source Spatio-Temporal Graph Neural Network Framework for Hand Trauma Risk Prediction"**  
Chen Ye, Xiaoqun Qin  
*PeerJ Computer Science* (Under Review), 2026

## 📋 Overview

This repository contains the complete implementation of STGNN‑HTR, a spatio-temporal graph neural network framework that integrates multi-source data for regional health risk forecasting. Key features:
- Hybrid graph construction (geographic adjacency + economic similarity)
- Graph convolutional networks for spatial modeling
- Gated recurrent units for temporal dynamics
- Multi-head attention for adaptive feature fusion
- GNNExplainer-based interpretability

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.3+

### Installation
```bash
git clone https://github.com/CIAM-Lab/stgnn-hand-injury.git
cd stgnn-hand-injury
pip install -r requirements.txt
Data Preparation
The raw data is available from the Hunan Provincial Health Commission (subject to data use agreements). The preprocessed dataset used in our experiments can be downloaded from: [Zenodo link - to be added after DOI creation]

Place the data in the following structure:
data/
├── hand_trauma_2011_2023.csv
└── city_features.csv
Running Experiments
Train the model:
python src/train.py --config configs/default.yaml
Reproduce paper results:
# Table 2: Baseline comparison
python scripts/run_baselines.py

# Table 3: Ablation studies
python scripts/run_ablation.py

# Figure 2: Sensitivity analysis
python scripts/run_sensitivity.py

# Interpretability analysis
python scripts/run_gnnexplainer.py
Results
The model achieves the following performance on the test set (2022-2023):
Metric	Value
MAE	0.365
RMSE	0.608
R²	0.851
Key findings:

Manufacturing employment ratio and physician density are the most influential predictors

Spatial dependencies extend beyond geographic adjacency to economic similarity

The Chang-Zhu-Tan urban agglomeration shows persistent high risk
Project Structure
stgnn-hand-injury/
├── src/               # Source code
│   ├── model.py       # STGNN‑HTR model definition
│   ├── train.py       # Training script
│   └── utils.py       # Utility functions
├── scripts/           # Experiment scripts
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks for visualization
├── data/              # Data directory (not included in repo)
├── requirements.txt   # Python dependencies
└── README.md          # This file
License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
Citation
If you use this code in your research, please cite:
@article{ye2026stgnn,
  title={STGNN‑HTR: An Open-Source Spatio-Temporal Graph Neural Network Framework for Hand Trauma Risk Prediction},
  author={Ye, Chen and Qin, Xiaoqun},
  journal={PeerJ Computer Science},
  year={2026},
  note={Under Review}
}
Contact
Chen Ye: 295256318@qq.com

Xiaoqun Qin: 1094426728@qq.com (Corresponding author)
Acknowledgments
This work was supported by the Provincial Health Research Program under Grant No. 25C1230. We thank the Hunan Provincial Health Commission and the Hunan Provincial Bureau of Statistics for providing the data.
