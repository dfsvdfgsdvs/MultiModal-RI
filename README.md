# MultiModal-RI

**System-Independent Retention Index Prediction via Multimodal Molecular Representation Learning**

MultiModal-RI is a deep learning framework that predicts liquid chromatography retention indices (RI) directly from molecular structures. It integrates a radial basis function Kolmogorov-Arnold network graph neural network (RBF-KAN GNN) with a custom MolBERT encoder through a bidirectional cross-attention fusion module, enabling accurate and system-independent RI prediction for untargeted metabolomics.

## Overview

Accurate metabolite annotation in untargeted metabolomics remains challenging due to the extensive chemical space and structural similarity of small molecules. Although liquid chromatography retention indices (RI) provide a stable and reproducible metric for compound identification, their application is limited by the sparse coverage of experimental RI databases. MultiModal-RI addresses this gap by predicting RI directly from molecular structures (SMILES strings), providing an orthogonal annotation dimension to mass spectrometry matching.

### Key Features

- **Dual-modal molecular representation**: 2D molecular graph topology (RBF-KAN GNN) + 1D SMILES sequence (MolBERT)
- **Bidirectional cross-attention fusion**: Dynamic alignment between graph and sequence modalities
- **RBF-KAN regressor**: Replaces traditional MLPs with learnable radial basis function activations
- **SMILES augmentation**: Randomized, non-canonical, kekulized, and isomeric SMILES variants for data augmentation
- **Dynamic dual-threshold filtering**: Error-scaled RI filtering window for metabolite annotation

### Performance

| Model | MAE | RMSE | R² | MedAE |
|-------|-----|------|----|-------|
| GNN-RT | 111.53 | 166.97 | 0.82 | 78.50 |
| KA-GCN | 78.98 | 135.67 | 0.90 | 46.04 |
| DeepGCN-RT | 84.36 | 164.28 | 0.83 | 50.04 |
| ABTMPNN | 76.36 | 151.61 | 0.84 | 42.02 |
| FPGNN | 92.89 | 155.11 | 0.86 | 56.54 |
| **MultiModal-RI** | **60.81** | **99.62** | **0.93** | **32.14** |

## Architecture

MultiModal-RI consists of four components:

1. **RBF-KAN GNN**: Encodes 2D molecular graph topology. Each graph convolutional layer computes attention coefficients via a parameterized RBF-KAN linear layer with inverse quadratic activation, followed by softmax normalization over the neighbor set.

2. **MolBERT Encoder**: A custom BERT-type transformer processes 1D SMILES strings. Input representations combine token embeddings, learnable positional encodings, and discrete embeddings for local chemical properties (atom type, bond presence, branching brackets, ring junction depth).

3. **Cross-Attention Fusion**: Bidirectional cross-attention exchanges contextual information between the graph topology vector and the sequence vector. Query, key, and value projections are computed for both modalities, and the aligned features are concatenated with residual connections.

4. **RBF-KAN Regressor**: Maps the fused multimodal descriptor to predicted RI. KAN layers place learnable non-linear univariate functions (parameterized as RBF kernels) on network connections rather than fixed activations at nodes, with compound regularization for smooth mapping.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- RDKit
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

The in-house dataset (3,951 compounds, reversed-phase LC, RI range 100–2400) used in this study is **not publicly available** due to proprietary restrictions. Researchers interested in obtaining the data for academic purposes should contact the corresponding authors.

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
| `layer_output` | 4 | Number of regressor layers |
| `batch_train` | 32 | Training batch size |
| `lr` | 1e-3 | Learning rate |
| `iteration` | 400 | Number of training epochs |
| `fusion_type` | 'cross_attention' | Fusion method ('cross_attention', 'gated', 'concat', 'attention', 'bilinear') |
| `use_augmentation` | True | Enable SMILES augmentation |
| `num_augments` | 4 | Number of augmented SMILES per molecule |
| `num_centers` | 6 | Number of RBF centers |
| `rbf_type` | 'inverse_quadratic' | RBF activation type |


### Prediction

To predict RI for new molecules, load the pretrained model:

```python
import torch
from kan_RBF import MultiModalRBFKAAGNN

# Load trained model
model = MultiModalRBFKAAGNN(
    N=10000, dim=96, layer_hidden=4, layer_output=4, vocab_size=vocab_size,
    bert_dim=512, bert_heads=16, bert_layers=8,
    num_centers=6, rbf_type='inverse_quadratic', heads=4,
    fusion_type='cross_attention', fusion_heads=6
)
model.load_state_dict(torch.load('data/inhouse_model_multimodal.h5'))
model.eval()

# Predict RI from SMILES
predicted_ri = model.forward_regressor(data_batch, train=False)
```



## Project Structure

```
MultiModal-RI/
├── MultiModal-RI.py       # Main training and evaluation script
├── kan_RBF.py             # Model definitions (RBF-KAN GNN, fusion, regressor)
├── molbert.py             # MolBERT encoder and SMILES tokenizer
├── preprocess.py          # Data preprocessing and SMILES augmentation
├── data/                  # Datasets and model checkpoints
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```


## License

This project is licensed under the MIT License.

## Acknowledgments

This work was financially supported by the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2025A1515012831) and the National Natural Science Foundation of China (Grant Nos. 32470685, 22361132526, 22274119).
