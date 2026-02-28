from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI(
    title="DNA Variant Pathogenicity Classifier",
    description="Predicts whether a genetic variant is pathogenic or benign",
    version="1.0.0"
)

# Load model
with open("models/classifier.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

class Variant(BaseModel):
    chromosome: str
    ref_allele: str
    alt_allele: str

def build_features(v: Variant):
    f = {}
    f['is_transition'] = 1 if set([v.ref_allele, v.alt_allele]) in [{'A','G'},{'C','T'}] else 0
    f['is_transversion'] = 1 - f['is_transition']
    for base in ['A','T','G','C']:
        f[f'ref_is_{base}'] = int(v.ref_allele == base)
        f[f'alt_is_{base}'] = int(v.alt_allele == base)
    c = v.chromosome.replace('chr','')
    f['chromosome_num'] = 23 if c=='X' else 24 if c=='Y' else 25 if c=='MT' else int(c) if c.isdigit() else 0
    f['is_sex_chrom']  = int(v.chromosome in ['X','Y'])
    f['is_autosome']   = int(v.chromosome not in ['X','Y','MT'])
    f['has_gene_symbol'] = 1
    return pd.DataFrame([f])[feature_names]

@app.get("/")
def root():
    return {"status": "healthy", "model": "DNA Variant Pathogenicity Classifier v1.0"}

@app.post("/predict")
def predict(variant: Variant):
    X    = build_features(variant)
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return {
        "chromosome":        variant.chromosome,
        "ref_allele":        variant.ref_allele,
        "alt_allele":        variant.alt_allele,
        "prediction":        "Pathogenic" if pred == 1 else "Benign",
        "confidence":        round(float(max(prob)), 4),
        "pathogenic_prob":   round(float(prob[1]), 4),
        "benign_prob":       round(float(prob[0]), 4),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
