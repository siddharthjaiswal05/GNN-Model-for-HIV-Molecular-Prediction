# HIV Molecular Activity Prediction using Graph Neural Networks

## Project Overview

This is a **state-of-the-art Graph Neural Network (GNN)** project for predicting whether molecules can inhibit HIV replication. The project uses the **Graph Isomorphism Network (GIN)** architecture, which is one of the most powerful GNN models available.
1. **Graph representation of molecules**: How to represent chemical compounds as graphs
2. **GNN architecture**: Understanding Graph Isomorphism Networks
3. **Complete ML pipeline**: Train-validation-test split, proper evaluation
4. **Preventing overfitting**: Dropout, batch normalization, early stopping
5. **Model evaluation**: ROC-AUC, precision-recall, confusion matrices
6. **Visualization**: Training curves, molecular graphs, prediction analysis
## Dataset Information
- **Source**: MoleculeNet benchmark (automatically downloaded via PyTorch Geometric)
- **Size**: ~41,000 molecules with experimental HIV activity labels
- **Task**: Binary classification (Active vs Inactive)
- **Format**: Graph data where:
  - **Nodes** = Atoms (with features like atom type, charge, etc.)
  - **Edges** = Chemical bonds (with bond type information)

### Why Molecules as Graphs?

Molecules are naturally represented as graphs:
```
    O          Node: Oxygen atom
    ||         Edge: Double bond
H - C - O - H  Nodes: Carbon, Oxygen, Hydrogen
    |          Edges: Single bonds
    H
```

Traditional ML methods (like Random Forests) can't capture this structural information effectively
## Model Architecture: GIN (Graph Isomorphism Network)

### Why GIN?

1. **Theoretically powerful**: As expressive as the Weisfeiler-Lehman graph isomorphism test
2. **Proven performance**: State of the-art on molecular benchmarks
3. **Learnable aggregation**: Better than simple mean/max pooling

### Architecture Details

```
Input Graph (Molecule)
    ↓
GIN Layer 1 (MLP + Aggregation)
    ↓
Batch Normalization + ReLU + Dropout
    ↓
GIN Layer 2-5 (Same structure)
    ↓
Global Pooling (Sum over all atoms)
    ↓
MLP Classifier
    ↓
Output (Active/Inactive)
```

### Key Components

1. **GIN Convolution**: 
   - Aggregates neighbor information
   - Learnable epsilon parameter
   - MLP transformation
   
2. **Batch Normalization**: 
   - Stabilizes training
   - Faster convergence
   
3. **Dropout (0.3)**: 
   - Prevents overfitting
   - Forces robust features
   
4. **Global Pooling**: 
   - Converts node features → graph features
   - We use sum pooling (best for GIN)

---

## Anti-Overfitting 

The project implements **multiple techniques** to prevent overfitting:

| Technique | How It Helps | Where Used |
|-----------|--------------|------------|
| **Dropout (0.3)** | Randomly drops neurons during training | After each GIN layer |
| **Batch Normalization** | Normalizes activations | After each GIN layer |
| **L2 Regularization** | Penalizes large weights | Weight decay in optimizer |
| **Early Stopping** | Stops when validation stops improving | Training loop (patience=15) |
| **Learning Rate Scheduling** | Reduces LR when plateau | ReduceLROnPlateau |
| **Gradient Clipping** | Prevents exploding gradients | Before optimizer step |
| **Train-Val-Test Split** | Proper evaluation | 70-15-15 split |
| **Class Weights** | Handles imbalanced data | Cross-entropy loss |

## Pipeline

### 1. Data Loading
```python
dataset = MoleculeNet(root='./data', name='HIV')
# Automatically downloads and processes the dataset
```

### 2. Data Split (70-15-15)
```python
train_dataset, val_dataset, test_dataset = create_data_splits(dataset)
# Stratified split maintains class distribution
```

### 3. Model Creation
```python
model = GINModel(
    num_node_features=dataset.num_node_features,
    hidden_dim=128,
    num_layers=5,
    dropout=0.3
)
```

### 4.Training Loop
- Train for 100 epochs (with early stopping)
- Validate after each epoch
- Save best model based on validation ROC-AUC
- Stop if no improvement for 15 epochs

### 5. Evaluation
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Classification Report


### Requirements

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit-pypi matplotlib seaborn scikit-learn pandas numpy tqdm
```

### Expected Output

- **Dataset statistics visualization**
- **Training history plots**
- **Evaluation metrics**
- **ROC and Precision-Recall curves**
- **Trained model saved as** `best_gin_model.pth`


## Results
- **Test Accuracy**: ~92.61%
- **Test ROC-AUC**: ~0.7870
- **Training time**: 10-30 minutes (CPU), 5-10 minutes (GPU)

### Understanding the Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Accuracy** | Overall correctness | 92.61% |
| **ROC-AUC** | Separability of classes | 0.7870 |
| **Precision** | Of predicted actives, how many are truly active | >0.98 |
| **Recall** | Of true actives, how many did we find | >0.94 |


## Project Structure

```
├── hiv_molecular_prediction_gnn.ipynb  # Main notebook
├── data/                               # Dataset (auto-downloaded)
├── dataset_statistics.png              # Data visualization
├── training_history.png                # Training curves
├── evaluation_results.png              # Test results
├── best_gin_model.pth                  # Saved model
└── README.md                           # This file
```

## Citation

If you use this project, please cite:

```bibtex
@misc{hiv_gnn_project,
  title={HIV Molecular Activity Prediction using Graph Neural Networks},
  author={Siddharth Jaiswal},
  year={2025},
  howpublished={\url{https://github.com/siddharthjaiswal05/GNN-Model-for-HIV-Molecular-Prediction}}
}
```

## Acknowledgments

- **PyTorch Geometric** team for the amazing library
- **MoleculeNet** for providing benchmark datasets
- **RDKit** for molecular processing tools
- **Graph Neural Network community** for research and tutorials
