# MultiModal-RI

**System-Independent Retention Index Prediction via Multimodal Molecular Representation Learning**

MultiModal-RI is a deep learning framework that predicts liquid chromatography retention indices (RI) directly from molecular structures. It integrates a radial basis function Kolmogorov-Arnold network graph neural network (RBF-KAN GNN) with a Transformer-based molecular language encoder (MolBERT) through a cross-attention fusion module, enabling accurate and system-independent RI prediction for untargeted metabolomics.

## Overview

Accurate metabolite annotation in untargeted metabolomics remains challenging due to the extensive chemical space and structural similarity of small molecules. While liquid chromatography retention indices provide a stable and reproducible metric for compound identification, their application is limited by the sparse coverage of experimental RI databases. MultiModal-RI addresses this gap by predicting RI directly from molecular structures (SMILES strings), providing an orthogonal annotation dimension to mass spectrometry matching.

### Key Features

- **Dual-modal molecular representation**: 2D molecular graph topology (RBF-KAN GNN) + 1D SMILES sequence (MolBERT)
- **Cross-attention fusion**: Bidirectional alignment between graph and sequence modalities
- **RBF-KAN regressor**: Replaces traditional MLPs with learnable radial basis function activations
- **SMILES augmentation**: Random, canonical, kekulized, and isomeric SMILES variants for data augmentation
- **Dynamic dual-threshold filtering**: Error-scaled RI filtering window for metabolite annotation

### Performance

| Metric | Value |
|--------|-------|
| MAE | 60.81 |
| RMSE | 99.62 |
| R² | 0.93 |
| MedAE | 32.14 |

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- RDKit
- DGL (Deep Graph Library)
- NumPy
- Scikit-learn
- Pandas
- Matplotlib

## Installation

```bash
git clone https://github.com/your-username/MultiModal-RI.git
cd MultiModal-RI
pip install -r requirements.txt
```

## Data Preparation

The in-house dataset (3,951 compounds, reversed-phase LC) used in this study is **not publicly available** due to proprietary restrictions. Researchers interested in obtaining the data for academic purposes should contact the corresponding authors.

To train MultiModal-RI on your own RI dataset, prepare the following files in the `data/` directory:

```
data/
├── train_set_stratified.txt    # Training set (SMILES + RI)
├── test_set_stratified.txt     # Test set (SMILES + RI)
├── atom_dict.pickle             # Atom feature dictionary
├── bond_dict.pickle             # Bond feature dictionary
├── edge_dict.pickle             # Edge feature dictionary
├── fingerprint_dict.pickle      # Molecular fingerprint dictionary
└── tokenizer_vocab.json         # SMILES tokenizer vocabulary
```

Each line in the training/test set text file should contain a SMILES string and the corresponding experimental RI value, tab-separated. The pickle dictionaries are automatically generated during the first run of `preprocess.py`.

## Usage

### Training

```bash
python MultiModal-RI.py
```

Key parameters can be modified in the `if __name__ == "__main__"` section of `MultiModal-RI.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 96 | Feature dimension |
| `layer_hidden` | 4 | Number of GNN hidden layers |
| `layer_output` | 4 | Number of GNN output layers |
| `batch_train` | 32 | Training batch size |
| `lr` | 1e-3 | Learning rate |
| `iteration` | 400 | Number of training epochs |
| `fusion_type` | 'gated' | Fusion method ('gated', 'cross_attention', 'concat', 'attention', 'bilinear') |
| `use_augmentation` | True | Enable SMILES augmentation |
| `num_augments` | 4 | Number of augmented SMILES per molecule |

### Pretrained Model

The trained model weights (`inhouse_model_multimodal.h5`) are available on HuggingFace:

- [https://huggingface.co/your-username/MultiModal-RI](https://huggingface.co/your-username/MultiModal-RI)

Download the model file and place it in the `data/` directory before running prediction.

### Prediction

To predict RI for new molecules, load the pretrained model:

```python
import torch
from kan_RBF import MultiModalRBFKAAGCN

# Load trained model
model = MultiModalRBFKAAGCN(...)
model.load_state_dict(torch.load('data/inhouse_model_multimodal.h5'))
model.eval()

# Predict RI from SMILES
predicted_ri = model.forward_regressor(data_batch, train=False)
```

### Metabolite Annotation with RI Filtering

MultiModal-RI predictions can be integrated as a secondary filter into metabolite annotation workflows (e.g., DeepMASS). The dual-threshold filtering window:
- Absolute error > 111 for RI < 700
- Relative error > 17% for RI ≥ 700

## Project Structure

```
MultiModal-RI/
├── MultiModal-RI.py       # Main training and evaluation script
├── kan_RBF.py             # Model definitions (RBF-KAN GNN, fusion, regressor)
├── molbert.py             # MolBERT encoder and SMILES tokenizer
├── preprocess.py          # Data preprocessing and augmentation
├── data/                  # Datasets and model checkpoints
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Citation

If you use MultiModal-RI in your research, please cite:

```bibtex
@article{wang2026multimodal,
  title={System-Independent Retention Index Prediction via Multimodal Molecular Representation Learning},
  author={Wang, Xujie and Ji, Hongchao and Zhu, Quanfei and Feng, Yuqi},
  journal={Analytical Chemistry},
  year={2026},
  publisher={American Chemical Society}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This work was financially supported by the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2025A1515012831) and the National Natural Science Foundation of China (Grant No. 32470685).
