# **HackNation 2025 — Challenge 9**  
## **Mini AlphaFold: Small-Scale Protein Structure & Drug Discovery AI**  
**Track:** VC Big Bets (Healthcare)  
**Section:** Small Model Deployment  
**Objective:** Ligand–Protein Binding Affinity Estimation — *Given a protein structure and a small molecule ligand, predict whether they bind strongly*  

---

**Project Title:** **Advanced AI-Powered Protein–Ligand Binding Prediction System**  
**Team:** Muhammed Nehan, Arish Shahab, Aaron Yu  

**GitHub Repository:** [muhnehh/hacknation2025-kinase-prediction](https://github.com/muhnehh/hacknation2025-kinase-prediction)  


## Project Overview

A comprehensive machine learning pipeline for predicting protein-ligand binding affinity and binary binding classification, developed for **HackNation 2025 Challenge 9**. The system combines molecular fingerprints with protein sequence embeddings using a three-model ensemble approach: baseline, fusion, and calibrated prediction models.

This project addresses the core challenge of **ligand-protein binding affinity estimation** by predicting whether a given protein structure and small molecule ligand bind strongly, implementing a production-ready solution optimized for the **Small Model Deployment** section of the healthcare track.

### Scientific Background

The system predicts binding affinity expressed as **pX values**, which represent the negative logarithm (base 10) of binding affinity in molar concentration. Higher pX values indicate stronger binding (e.g., pX = 7.0 corresponds to 100 nM binding affinity).

### HackNation Challenge Solution

Our solution directly addresses **Challenge 9's objective** by:
- **Input**: Protein sequences and small molecule SMILES structures
- **Output**: Binding probability predictions with confidence scores
- **Innovation**: Multi-model ensemble with temperature calibration for reliable uncertainty estimation
- **Deployment**: Optimized 23MB model size suitable for edge deployment
- **Interface**: Production-ready web application with real-time predictions

## Key Features

- **Multi-model ensemble approach** for robust predictions
- **Molecular fingerprint generation** using ECFP4 (Extended Connectivity Fingerprints)
- **Protein sequence embedding** using ESM2 transformer model
- **Temperature scaling** for probability calibration
- **Real-time web interface** with scientific analysis capabilities
- **Comprehensive evaluation** with multiple metrics
- **Modular architecture** supporting both training and inference

### Web Interface Features
- 🎯 **Real-time Predictions**: Get binding predictions in under 120ms
- 🧬 **Scientific Analysis**: Comprehensive molecular property analysis
- 📊 **Interactive Metrics**: Clickable evaluation plots and performance dashboards
- 💊 **Optimization Suggestions**: Evidence-based recommendations for drug discovery
- 🔬 **Batch Processing**: Handle multiple predictions efficiently
- 📈 **Data Visualization**: Interactive charts and molecular property displays

## Architecture Overview

### Three-Model Pipeline

1. **Baseline Model**: ECFP4 molecular fingerprints processed through logistic regression
   - Fast inference and interpretable results
   - Uses 2048-bit molecular fingerprints with radius 2
   - Serves as performance baseline and fallback option

2. **Fusion Model**: Combined molecular and protein features through neural network
   - Concatenates ECFP4 fingerprints with ESM2 protein embeddings
   - Multi-layer perceptron with dropout regularization
   - Simultaneous binary classification and regression prediction

3. **Calibrated Model**: Temperature scaling applied to fusion model outputs
   - Improves probability calibration for better uncertainty estimation
   - Uses validation data to learn optimal temperature parameter
   - Provides well-calibrated confidence scores

### Technical Implementation

- **Molecular Features**: ECFP4 fingerprints (2048 dimensions)
- **Protein Features**: ESM2-t6-8M embeddings (320 dimensions)
- **Optimization**: AdamW optimizer with gradient clipping
- **Training**: Multi-task learning with classification and regression heads
- **Evaluation**: Comprehensive metrics including AUROC, AUPRC, and calibration scores

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, CPU training supported)
- Minimum 8GB RAM (16GB recommended)
- Node.js 16+ (for web interface)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/muhnehh/hacknation2025-kinase-prediction.git
cd hacknation2025-kinase-prediction
```

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch, rdkit, transformers; print('Dependencies installed successfully')"
```

### 4. Web Interface Setup
```bash
cd web
npm install
```

## Quick Start

### 🚀 Try the Web Interface
1. **Live Demo**: Visit our deployed application (link above)
2. **Select a Target**: Choose from kinase proteins (e.g., SRC_HUMAN)
3. **Input Molecule**: Enter a SMILES string or use examples
4. **Get Predictions**: Receive binding probability and scientific analysis instantly

### Training Pipeline
```bash
# Train the complete pipeline
python train.py

# Test trained models
python test_final_models.py

# Run interactive predictions
python predict.py
```

### Web Application (Local Development)
```bash
# Start API server (Terminal 1)
python api_server.py

# Start web interface (Terminal 2)
cd web
npm run dev
```

## Core Dependencies

### Machine Learning Stack
```
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.20.0
scikit-learn>=1.1.0
```

### Chemistry & Molecular Processing
```
rdkit>=2022.03.0
```

### Data Processing & Visualization
```
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
```

### Web Framework
```
fastapi>=0.68.0
uvicorn>=0.15.0
```

## Project Structure

```
hacknation2025/
├── train.py                           # Main training pipeline
├── api_server.py                      # FastAPI backend server
├── predict.py                         # Prediction interface
├── test_final_models.py              # Model evaluation
├── step1compare.py                    # Baseline model comparison
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── saved_models/                      # Trained model artifacts
│   ├── baseline_logreg.pkl           # Baseline logistic regression
│   ├── fusion_mlp.pth                # Fusion neural network
│   ├── temperature_scaler.pth        # Calibration model
│   └── protein_cache.pkl             # Cached protein embeddings
├── web/                              # Next.js web interface
│   ├── app/                          # App router pages
│   ├── components/                   # React components
│   ├── lib/                         # Utility functions
│   └── public/                      # Static assets
└── lightning_logs/                   # Training logs and metrics
```

## Data Format

The system expects CSV files with the following columns:

- `target_entry`: Protein identifier
- `sequence`: Amino acid sequence
- `smiles`: Ligand SMILES string
- `px`: Binding affinity (negative log10 molar)
- `label`: Binary classification (0 or 1)

**Default training files:**
- `bindingdb_kinase_top10_train.csv`
- `bindingdb_kinase_top10_val.csv`
- `bindingdb_kinase_top10_test.csv`

## Performance Metrics

### Model Performance
- **Predictive Accuracy**: 82% AUROC
- **Inference Speed**: 120ms average latency
- **Reliability Score**: 93% calibration quality

### Detailed Metrics
- **AUROC**: 0.82 (exceeds industry benchmark of 0.75)
- **PR-AUC**: 0.70
- **Calibration (ECE)**: 0.07
- **pX prediction R²**: 0.60-0.80
- **Model size**: 23MB (optimized for deployment)

## Visual Demonstrations

### Web Interface Screenshots

![Web Interface Demo](web_interface_demo.png)
*Complete web interface demonstration showing real-time prediction capabilities and scientific analysis*

![Web Interface - Prediction Results](web/public/demo-prediction.png)
*Real-time prediction interface with scientific analysis capabilities*

![Metrics Dashboard](web/public/demo-metrics.png)
*Comprehensive performance metrics and model evaluation dashboard*

### Model Evaluation Results

![Final Model Evaluation](final_model_evaluation.png)
*Comprehensive model evaluation including ROC curves, precision-recall analysis, and calibration plots*

![Training Curves](saved_models/training_curves.png)
*Training dynamics showing loss convergence and validation performance over epochs*

### Web Interface Capabilities

**Scientific Analysis Output Example:**
- **Binding Probability**: 97.3% (High confidence)
- **Molecular Weight**: 604.64 Da
- **LogP**: 6.2 (Lipophilicity)
- **TPSA**: 114.52 Ų (Polar surface area)
- **Rotatable Bonds**: 7
- **Drug-likeness Score**: 0.54
- **Optimization Suggestions**: 
  - Reduce molecular weight (<500 Da) for oral bioavailability
  - Decrease lipophilicity (LogP <5) by adding polar groups
  - Balance molecular properties using medicinal chemistry principles

## API Endpoints

### Prediction API
```bash
POST /predict
{
  "sequence": "MGSNKSKPKDAS...",
  "smiles": "CCN(CC)CCN(C)C(=O)..."
}
```

### Scientific Analysis API
```bash
POST /explain
{
  "sequence": "MGSNKSKPKDAS...",
  "smiles": "CCN(CC)CCN(C)C(=O)..."
}
```

## Usage Examples

### Programmatic Prediction
```python
from predict import predict_binding

sequence = "MGSNKSKPKDAS..."  # Protein sequence
smiles = "CCN(CC)CCN(C)C(=O)..."  # SMILES string

result = predict_binding(sequence, smiles)
print(f"Binding probability: {result['calibrated_probability']:.3f}")
print(f"Predicted pX: {result['predicted_px']:.2f}")
```

### Batch Processing
```python
import pandas as pd
from predict import batch_predict_binding

df = pd.read_csv('new_compounds.csv')
results = batch_predict_binding(
    sequences=df['sequence'].tolist(),
    smiles=df['smiles'].tolist()
)
```

## Configuration

### Training Parameters
```python
# Model architecture
BATCH_SIZE = 32
LR = 2e-3
EPOCHS = 32
PROT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Feature generation
ECFP_BITS = 2048
ECFP_RADIUS = 2
```

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB free space
- **GPU**: Optional (CUDA-compatible)

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM

## Troubleshooting

### Installation Issues
```bash
# RDKit installation problems
conda install rdkit -c conda-forge

# PyTorch CUDA compatibility
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Optimization
```python
# Reduce batch size for limited memory
BATCH_SIZE = 16  # or 8

# Use CPU-only mode
DEVICE = "cpu"
```

## Team

**HackNation 2025 Team:**
- **Muhammed Nehan** - Lead Developer & ML Engineer
- **Arish Shahab** - Data Scientist & Model Architecture
- **Aaron Yu** - Full-Stack Developer & UI/UX

## Acknowledgments

- **BindingDB**: Comprehensive binding affinity database
- **Meta AI**: ESM2 protein language model
- **RDKit**: Open-source molecular informatics toolkit
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning library
- **HackNation 2025**: For providing the platform and opportunity

## Repository

🔗 **GitHub Repository:** https://github.com/muhnehh/hacknation2025-kinase-prediction

## License

This project is released under the MIT License.

---

**HackNation 2025 Project** | Advanced Protein-Ligand Binding Prediction System  
**Challenge 9:** Mini AlphaFold – Small-Scale Protein Structure & Drug Discovery AI  
**Track:** VC Big Bets (Healthcare) | **Section:** Small Model Deployment  
**Team:** Muhammed Nehan, Arish Shahab, Aaron Yu


