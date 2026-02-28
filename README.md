# 🧬 DNA Variant Pathogenicity Classifier

**Author:** Rosemary James · ML/Bioinformatics Platform Engineer  
**Stack:** Python · Scikit-learn · FastAPI · BioPython · ClinVar · Random Forest  
**Status:** Production-ready  
**ROC-AUC:** 0.7581

---

## What This Does

A machine learning classifier that predicts whether a human genetic variant is **Pathogenic** or **Benign** - trained on real clinical data from the NCBI ClinVar database (8.8 million variants).

```
Genetic Variant (chromosome + ref + alt)
    │
    ▼
Feature Engineering (biological features)
    │
    ▼
Random Forest Classifier (trained on ClinVar)
    │
    ▼
Pathogenicity Prediction + Confidence Score
    │
    ▼
REST API Response (FastAPI)
```

---

## Results

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| **Random Forest** | **0.7581** 🏆 | **69%** |
| Gradient Boosting | 0.7303 | 67% |
| Logistic Regression | 0.5918 | 58% |

**Top predictive features:**
- Chromosome number (75% importance) - certain chromosomes are mutation hotspots
- Nucleotide identity - which base is changing
- Transition vs transversion - type of nucleotide change

---

## Quick Start

### Install dependencies
```bash
pip install scikit-learn fastapi uvicorn pandas numpy requests biopython
```

### Prepare data (downloads real ClinVar data ~100MB)
```bash
python3 data/prepare_data.py
```

### Train the model
```bash
python3 models/train.py
```

### Start the API
```bash
python3 api/serve.py
```

### Make a prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"chromosome": "17", "ref_allele": "A", "alt_allele": "G"}'
```

**Response:**
```json
{
  "chromosome": "17",
  "ref_allele": "A",
  "alt_allele": "G",
  "prediction": "Pathogenic",
  "confidence": 0.5714,
  "pathogenic_prob": 0.5714,
  "benign_prob": 0.4286
}
```

---

## Project Structure

```
dna-variant-classifier/
├── data/
│   ├── prepare_data.py        # Download + filter ClinVar dataset
│   └── clinvar_prepared.csv   # Processed dataset (5,000 balanced variants)
├── models/
│   ├── train.py               # Train + evaluate classifier
│   ├── classifier.pkl         # Saved Random Forest model
│   ├── feature_names.pkl      # Feature names for serving
│   └── metrics.csv            # Model performance metrics
└── api/
    └── serve.py               # FastAPI prediction endpoint
```

---

## API Reference

### `GET /`
Health check
```json
{"status": "healthy", "model": "DNA Variant Pathogenicity Classifier v1.0"}
```

### `POST /predict`
Predict pathogenicity of a variant

**Request:**
```json
{
  "chromosome": "17",
  "ref_allele": "A",
  "alt_allele": "G"
}
```

**Response:**
```json
{
  "chromosome": "17",
  "ref_allele": "A",
  "alt_allele": "G",
  "prediction": "Pathogenic",
  "confidence": 0.5714,
  "pathogenic_prob": 0.5714,
  "benign_prob": 0.4286
}
```

---

## The Data

**ClinVar** is NCBI's database of human genetic variants and their clinical significance. It contains variants submitted by clinical labs, research institutions, and pharmaceutical companies worldwide.

- 8,889,525 total variants downloaded
- Filtered to single nucleotide variants (SNVs) with clear Pathogenic/Benign labels
- Balanced to 2,500 pathogenic + 2,500 benign = 5,000 training samples
- 80/20 train/test split

---

## Key Concepts

| Term | What it means |
|------|--------------|
| SNV | Single Nucleotide Variant - one DNA base changed |
| Pathogenic | Variant causes or contributes to disease |
| Benign | Variant has no disease effect |
| ROC-AUC | Area Under ROC Curve - 0.5 = random, 1.0 = perfect |
| Transition | Purine↔Purine or Pyrimidine↔Pyrimidine change (A↔G, C↔T) |
| Transversion | Purine↔Pyrimidine change (A↔C, A↔T, G↔C, G↔T) |

---

## Related Projects

This is Phase 2 of a 5-project bioinformatics ML portfolio:

| Phase | Project | Status |
|-------|---------|--------|
| 1 | [Genomics Variant Calling Pipeline](https://github.com/roseemaryy/genomics-variant-pipeline) | ✅ Complete |
| 2 | **DNA Variant Pathogenicity Classifier** (this repo) | ✅ Complete |
| 3 | Single-Cell RNA Cell Type Classifier (Scanpy + scVI) | 🔨 Building |
| 4 | Drug Toxicity Predictor (DeepChem + PyTorch) | 📋 Planned |
| 5 | Bioinformatics MLOps Platform (unified platform) | 📋 Planned |

---

## Author

**Rosemary James** · ML/Bioinformatics Platform Engineer  
📧 nyamburaromy@gmail.com  
🔗 [GitHub](https://github.com/roseemaryy)  