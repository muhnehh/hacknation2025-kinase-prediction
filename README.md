# 🧬 Mini AlphaFold: Protein–Ligand Binding AI  

<div align="center">

[![HackNation](https://img.shields.io/badge/HackNation-2025-00D9FF?style=for-the-badge&logo=hackathon&logoColor=white)](https://hacknation.ca)
[![Challenge 9](https://img.shields.io/badge/Challenge-9-FF6B6B?style=for-the-badge)](https://hacknation.ca)
[![Track Healthcare](https://img.shields.io/badge/Track-Healthcare-4ECDC4?style=for-the-badge)](https://hacknation.ca)
[![Model Size 23MB](https://img.shields.io/badge/Model_Size-23MB-45B7D1?style=for-the-badge)](https://github.com/muhnehh/hacknation2025-kinase-prediction)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**🏥 VC Big Bets (Healthcare) | 🚀 Small Model Deployment**

### Quick Links
[📚 Documentation](#overview) · [🚀 Quick Start](#quick-start) · [📊 Performance](#performance) · [🔗 Repository](https://github.com/muhnehh/hacknation2025-kinase-prediction)

</div>

---

## 🎯 Objective

Building **production-ready AI** for accelerating drug discovery:

| Goal | Status | Details |
|------|--------|---------|
| 🧪 Predict Ligand-Protein Binding Affinity | ✅ Complete | Regression: pX values with R² = 0.80 |
| ⚡ Classify Binding Probability | ✅ Complete | Classification: AUROC = 0.82, PR-AUC = 0.70 |
| ⏱️ Real-time Inference | ✅ Complete | ~120 ms per prediction |
| 📦 Deployment Ready | ✅ Complete | 23 MB model size (edge-optimized) |

---

## 🎨 Vision

<table>
<tr>
<td width="50%">

### 💡 What We Built
A **full-stack machine learning system**:
- 🧪 **Molecular Analysis**: ECFP4 fingerprints for ligand representation
- 🧠 **Protein Intelligence**: ESM2 transformer embeddings (320-D)
- 🤖 **Ensemble Pipeline**: 3-model architecture for robustness
- 📊 **Calibration Engine**: Temperature scaling for reliable uncertainty
- 🌐 **Web Interface**: Interactive UI with real-time predictions
- 🔌 **REST API**: FastAPI backend for integration

</td>
<td width="50%">

### 🏥 Why It Matters

**Healthcare Impact:**
- 💊 **Accelerate Drug Discovery**: Screen 10,000s of compounds instantly
- 🏥 **Reduce Lab Costs**: 50-80% cost reduction in early screening
- 📱 **Mobile Deployment**: Works on edge devices & cloud
- ✅ **Production Ready**: Battle-tested metrics & calibration
- 🔬 **Scientific Rigor**: Based on ESM2 & ECFP4 (proven methods)
- 🌍 **Accessible**: Open-source, MIT license

</td>
</tr>
</table>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Protein & Ligand                  │
│         (Sequence: "MGSNKSKP..." | SMILES: "CCN(CC)...")    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   ┌─────────────┐              ┌──────────────────┐
   │   LIGAND    │              │   PROTEIN        │
   │  Encoder    │              │   Encoder        │
   │  (ECFP4)    │              │   (ESM2-t6-8M)   │
   │  2048-bit   │              │   320-D embeddings
   └──────┬──────┘              └────────┬─────────┘
          │                              │
          │  Fingerprints                │  Embeddings
          │  (2048 features)             │  (320 features)
          │                              │
          └────────────────┬─────────────┘
                           │
                ┌──────────┴───────────┐
                │                      │
                ▼                      ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Baseline Model  │  │  Fusion Model    │
        │  (Logistic Reg)  │  │  (Neural Network)│
        │  Fast & Simple   │  │  High Accuracy   │
        └────────┬─────────┘  └────────┬─────────┘
                 │                     │
                 │   Ensemble Vote     │
                 │                     │
                 └─────────┬───────────┘
                           │
                        ▼
            ┌────────────────────────────┐
            │  Calibrated Model          │
            │  (Temperature Scaling)     │
            │  Reliable Confidence       │
            └────────────┬───────────────┘
                         │
                    ▼
        ┌──────────────────────────┐
        │  PREDICTIONS & INSIGHTS  │
        │  • Binding Probability   │
        │  • pX (Affinity)         │
        │  • Confidence Score      │
        │  • Drug Likeness         │
        └──────────────────────────┘
```

### 🔬 Three-Model Pipeline

| # | Model | Method | Purpose |
|---|-------|--------|---------|
| 1️⃣ | **Baseline** | Logistic Regression + ECFP4 | Fast baseline, interpretable predictions |
| 2️⃣ | **Fusion** | Neural Network + Multi-task | High accuracy, learns complex patterns |
| 3️⃣ | **Calibrated** | Temp. Scaling | Reliable confidence scores |

---

## 📊 Technical Specifications

### 🔧 Core Components

<table>
<tr>
<td width="50%">

#### Molecular Features
| Component | Specification |
|-----------|--------------|
| **Ligand Encoder** | ECFP4 (Radius = 2) |
| **Fingerprint Size** | 2048-bit |
| **Fingerprint Type** | Extended Connectivity |
| **Hash Function** | Morgan algorithm |

#### Protein Features
| Component | Specification |
|-----------|--------------|
| **Protein Encoder** | ESM2-t6-8M |
| **Embedding Dim** | 320-D |
| **Architecture** | Transformer |
| **Training Data** | Uniref90 |

</td>
<td width="50%">

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-3 |
| **Batch Size** | 32 |
| **Gradient Clipping** | Yes |
| **Loss Function** | Multi-task (CE + MSE) |

#### Performance Metrics
| Metric | Score |
|--------|-------|
| **AUROC** | 🥇 0.82 |
| **PR-AUC** | 🥇 0.70 |
| **R² (pX)** | 🥇 0.80 |
| **ECE** | 0.07 (calibrated) |
| **Inference** | ~120 ms/pred |
| **Model Size** | 23 MB |

</td>
</tr>
</table>

---

## Installation

```bash
git clone https://github.com/muhnehh/hacknation2025-kinase-prediction.git
cd hacknation2025-kinase-prediction
pip install -r requirements.txt
```

### Training & Inference
```bash
python train.py
python test_final_models.py
python predict.py
```

### Web Interface
```bash
python api_server.py
cd web
npm run dev
```

---

## 🚀 Quick Start

### 📥 Installation

```bash
# Clone repository
git clone https://github.com/muhnehh/hacknation2025-kinase-prediction.git
cd hacknation2025-kinase-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 💻 Training & Inference

```bash
# Train models
python train.py

# Evaluate on test set
python test_final_models.py

# Make predictions
python predict.py
```

### 🌐 Web Interface & API

```bash
# Start FastAPI backend (runs on http://localhost:8000)
python api_server.py

# In another terminal, start Next.js frontend (http://localhost:3000)
cd web
npm install
npm run dev
```

### 🔧 API Quick Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "protein_sequence": "MGSNKSKPKDAS...",
    "smiles": "CCN(CC)CCN(C)C(=O)...",
    "use_calibration": true
  }'
```  

---

## 🖼️ Visual Demonstrations

### 📱 Web Interface Features

<img width="1616" height="1414" alt="Web Interface - Real-time Predictions" src="https://github.com/user-attachments/assets/cf35a464-6d6c-48ed-852e-e69a9bcee340" />

**✨ Features Shown:**
- ✅ Real-time molecular structure visualization
- ✅ Interactive binding affinity predictions
- ✅ Drug-likeness & molecular property analysis
- ✅ Confidence scores with uncertainty calibration
- ✅ Clean, modern UI built with Next.js

### 📈 Model Performance Results

```
🎯 KEY METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AUROC (Binding Classification):     0.82 ██████████████░ 
  PR-AUC (Precision-Recall):          0.70 ███████████░░░░
  R² Score (Affinity Regression):     0.80 ████████████░░░
  Expected Calibration Error (ECE):   0.07 ░░░░░░░░░░░░░░░
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏱️  Inference Speed:                120 ms per prediction
  📦 Model Size:                      23 MB (edge-ready)
```

---

## 📝 Data Format

### CSV Dataset Structure

```csv
target_entry,sequence,smiles,px,label
ALK,MGSNKSKPKDAS...,CCN(CC)CCN...,7.5,1
ALK,MGSNKSKPKDAS...,CCN(C)C(=O)...,6.2,1
BRAF,MEEPFYG...,CC(C)Cc1c...,5.1,0
```

| Column | Type | Description |
|--------|------|-------------|
| `target_entry` | String | Protein/Kinase identifier (UniProt ID) |
| `sequence` | String | Amino acid sequence (canonical) |
| `smiles` | String | SMILES notation for ligand |
| `px` | Float | Binding affinity (−log₁₀ Kd) |
| `label` | Int | Binary binding (1=binds strongly, 0=weak) |

### 📦 Default Datasets

```
bindingdb_kinase_top10_train.csv  → Training set (70%)
bindingdb_kinase_top10_val.csv    → Validation set (15%)
bindingdb_kinase_top10_test.csv   → Test set (15%)
```

Source: [BindingDB](https://www.bindingdb.org/) - Curated kinase binding data

---

## 🎯 Performance Breakdown

### Classification Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **AUROC** | 0.82 | Strong discrimination between binders & non-binders |
| **PR-AUC** | 0.70 | Good precision-recall trade-off (important for imbalanced data) |
| **Specificity** | 0.78 | Correctly identifies weak binders |
| **Sensitivity** | 0.86 | Identifies strong binders reliably |

### Regression Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **R² Score** | 0.80 | Model explains 80% of pX variance |
| **RMSE** | 0.65 pX | ~63% accuracy in affinity prediction |
| **MAE** | 0.52 pX | Average absolute error |

### Calibration Quality

| Metric | Value | Status |
|--------|-------|--------|
| **Expected Calibration Error** | 0.07 | ✅ Well-calibrated |
| **Confidence vs Accuracy** | Aligned | ✅ Predictions are reliable |

### Inference Efficiency

| Benchmark | Value | Device |
|-----------|-------|--------|
| **Per-Prediction** | ~120 ms | GPU (NVIDIA) |
| **Batch (32 samples)** | ~3.8 sec | GPU |
| **Throughput** | 8-10 pred/sec | CPU (multi-threaded) |
| **Memory Footprint** | ~2.1 GB | Runtime |

---

## 🔌 API Documentation

### Endpoints

#### POST /predict

**Request:**
```json
{
  "protein_sequence": "MGSNKSKPKDAS...",
  "smiles": "CCN(CC)CCN(C)C(=O)...",
  "use_calibration": true,
  "return_details": true
}
```

**Response:**
```json
{
  "binding_probability": 0.87,
  "predicted_px": 7.3,
  "confidence_score": 0.92,
  "drug_likeness": {
    "mw": 245.3,
    "logp": 2.1,
    "h_donors": 1,
    "h_acceptors": 4,
    "violations": 0,
    "lipinski_pass": true
  },
  "inference_time_ms": 118
}
```

#### POST /batch_predict

Process multiple predictions in one request.

#### GET /metrics

Retrieve model performance metrics and calibration stats.

---

## 💡 Code Examples

### Python Integration

```python
from predict import predict_binding

# Example protein sequence
protein_sequence = """MGSNKSKPKDASKKAESGEVSEKPSKSTPPKK
DLDSRLVDPPVDGEFLVDKVTKVGTLDSEVAV
VVDGTRGTPEDLEYFENTKKNFTYDTSNDVTL"""

# Example ligand (SMILES)
smiles = "CCN(CC)CCN(C)C(=O)c1ccc2c(c1)CC(C)N2"

# Make prediction
result = predict_binding(protein_sequence, smiles, use_calibration=True)

print(f"🎯 Binding Probability: {result['binding_probability']:.2%}")
print(f"📊 Predicted pX: {result['predicted_px']:.2f}")
print(f"📈 Confidence: {result['confidence_score']:.2%}")

# Check drug-likeness
druglike = result['drug_likeness']
print(f"\n💊 Lipinski's Rule of 5: {'PASS ✅' if druglike['lipinski_pass'] else 'FAIL ❌'}")
print(f"   Molecular Weight: {druglike['mw']:.1f} g/mol")
print(f"   LogP: {druglike['logp']:.2f}")
```

### Batch Processing

```python
from predict import batch_predict

# Load your data
import pandas as pd
data = pd.read_csv('compounds.csv')  # Must have 'sequence' and 'smiles' columns

# Predict on all
predictions = batch_predict(
    sequences=data['sequence'].tolist(),
    smiles_list=data['smiles'].tolist(),
    batch_size=32,
    use_calibration=True
)

# Add predictions to dataframe
data['binding_prob'] = [p['binding_probability'] for p in predictions]
data['px_pred'] = [p['predicted_px'] for p in predictions]

data.to_csv('predictions_output.csv', index=False)
```

---

## 🛠️ Technology Stack

<table>
<tr>
<td width="33%">

### 🤖 ML & AI
- **PyTorch** 2.0+
- **Transformers** (HuggingFace)
- **ESM2** (Meta AI)
- **scikit-learn**
- **RDKit**

</td>
<td width="33%">

### 🌐 Backend
- **FastAPI**
- **Uvicorn**
- **Pydantic**
- **Python 3.9+**

</td>
<td width="33%">

### 🎨 Frontend
- **Next.js 14**
- **React 18**
- **TypeScript**
- **Tailwind CSS**
- **Shadcn/ui**

</td>
</tr>
</table>

---

## 📚 Key Features

### ✨ Intelligent Predictions
- 🎯 Dual predictions: Classification + Regression
- 📊 Ensemble voting for robustness
- 🔐 Calibrated confidence scores
- ⚡ Fast inference (~120ms)

### 🔬 Scientific Analysis
- 💊 Drug-likeness assessment (Lipinski's Rule of 5)
- 📈 Molecular property analysis
- 🧮 Multi-task learning (classification + regression)
- 🔍 Explainability insights

### 🌐 Production Ready
- 🚀 REST API for easy integration
- 📱 Interactive web interface
- 📦 Tiny model size (23 MB)
- 🔌 Batch processing support
- 📊 Real-time metrics dashboard  

---

## 👥 Team

| Name | Role | Affiliation |
|------|------|-------------|
| **Muhammed Nehan** | 💻 Lead Developer & Data Science | |
| **Arish Shahab** | 🔬 Researcher | Harvard MS, Biomed @McMaster |
| **Aaron Yu** | 🧬 Research & Bioinformatics | OICR, Biomed @McMaster |

---

## 📄 License & Attribution

<div align="center">

**MIT License © 2025**

This project is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

**HackNation 2025 · Challenge 9**  
*Mini AlphaFold: Small-Scale Protein & Drug Discovery AI*

🏆 **VC Big Bets (Healthcare) Track**  
🚀 **Small Model Deployment Category**

</div>

---

## 🔗 Resources & References

### Datasets
- 📊 [BindingDB](https://www.bindingdb.org/) - Open drug binding database
- 🧬 [UniProt](https://www.uniprot.org/) - Protein knowledge base

### Models & Methods
- 🧠 [ESM2 Transformers](https://github.com/facebookresearch/esm) - Protein language model
- 🧪 [ECFP Fingerprints](https://www.rdkit.org/) - Molecular fingerprinting
- 🤖 [PyTorch](https://pytorch.org/) - Deep learning framework

### Related Work
- 📖 [AlphaFold2](https://www.deepmind.com/research/alphafold)
- 📖 [Protein-Ligand Docking](https://autodock.scripps.edu/)
- 📖 [SMILES Notation](https://en.wikipedia.org/wiki/Simplified_molecular_input_line_entry_system)

---

## 🚀 Future Enhancements

- [ ] 3D protein structure integration (RoseTTAFold)
- [ ] Ensemble with graph neural networks
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Real-time compound screening dashboard
- [ ] Multi-target prediction support
- [ ] Active learning for iterative improvement

---

## ❓ FAQ

**Q: What's the minimum hardware required?**  
A: CPU inference works fine for single predictions. GPU (NVIDIA RTX 3060+) recommended for batches.

**Q: Can I use this commercially?**  
A: Yes! MIT license allows commercial use. Just include the license notice.

**Q: How do I cite this work?**  
A: Use the citation format in [CITATION.cff](CITATION.cff).

**Q: What's the accuracy on your datasets?**  
A: AUROC=0.82, PR-AUC=0.70 on BindingDB kinase subset. Results vary by target.

---

<div align="center">

### ⭐ If this project helped you, please star it on GitHub!

[🔗 GitHub Repository](https://github.com/muhnehh/hacknation2025-kinase-prediction)

**Built with 💜 for HackNation 2025**

</div>



