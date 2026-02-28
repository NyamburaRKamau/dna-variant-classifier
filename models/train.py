import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Engineer real biological features from ClinVar data"""
    print("🔬 Engineering biological features...")
    
    features = pd.DataFrame()
    
    # 1. Variant type features
    features['is_transition'] = df.apply(
        lambda r: 1 if set([r['ReferenceAllele'], r['AlternateAllele']]) 
        in [{'A','G'}, {'C','T'}] else 0, axis=1
    )
    features['is_transversion'] = 1 - features['is_transition']
    
    # 2. Nucleotide identity
    for base in ['A', 'T', 'G', 'C']:
        features[f'ref_is_{base}'] = (df['ReferenceAllele'] == base).astype(int)
        features[f'alt_is_{base}'] = (df['AlternateAllele'] == base).astype(int)
    
    # 3. Chromosome features
    def chrom_to_num(c):
        c = str(c).replace('chr','')
        if c == 'X': return 23
        if c == 'Y': return 24
        if c == 'MT': return 25
        try: return int(c)
        except: return 0
    
    features['chromosome_num'] = df['Chromosome'].apply(chrom_to_num)
    features['is_sex_chrom']   = df['Chromosome'].isin(['X','Y']).astype(int)
    features['is_autosome']    = (~df['Chromosome'].isin(['X','Y','MT'])).astype(int)
    
    # 4. Gene features
    features['has_gene_symbol'] = (~df['GeneSymbol'].isna()).astype(int)
    
    print(f"   Features engineered: {features.shape[1]} features")
    return features

def train():
    print("🧬 Loading ClinVar dataset...")
    
    # Reload raw data with more columns
    import requests, gzip, io
    print("   Downloading fresh ClinVar data with full columns...")
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    response = requests.get(url, stream=True)
    
    with gzip.open(io.BytesIO(response.content), 'rt') as f:
        df = pd.read_csv(f, sep='\t', low_memory=False)
    
    # Filter to SNVs with clear labels
    snvs = df[
        (df['Type'] == 'single nucleotide variant') &
        (df['ClinicalSignificance'].isin(['Pathogenic', 'Benign'])) &
        (df['ReferenceAlleleVCF'].str.len() == 1) &
        (df['AlternateAlleleVCF'].str.len() == 1)
    ].copy()
    
    snvs['label']           = (snvs['ClinicalSignificance'] == 'Pathogenic').astype(int)
    snvs['ReferenceAllele'] = snvs['ReferenceAlleleVCF']
    snvs['AlternateAllele'] = snvs['AlternateAlleleVCF']
    
    # Balance
    pathogenic = snvs[snvs['label'] == 1].head(2500)
    benign     = snvs[snvs['label'] == 0].head(2500)
    snvs       = pd.concat([pathogenic, benign]).sample(frac=1, random_state=42)
    
    print(f"   {len(snvs):,} variants loaded")
    
    # Engineer features
    X = engineer_features(snvs)
    y = snvs['label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train multiple models
    models = {
        'Random Forest':    RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boost':   GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    best_auc   = 0
    best_model = None
    best_name  = ""
    
    print("\n🏋️  Training models...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        print(f"   {name}: ROC-AUC = {auc:.4f}")
        
        if auc > best_auc:
            best_auc   = auc
            best_model = clf
            best_name  = name
    
    print(f"\n🏆 Best model: {best_name} (ROC-AUC: {best_auc:.4f})")
    
    print("\n📊 Full evaluation:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred,
          target_names=['Benign', 'Pathogenic']))
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature':    X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n🔬 Top features:")
        print(importance.head(8).to_string(index=False))
    
    # Save
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    pd.DataFrame([{
        "model":   best_name,
        "roc_auc": round(best_auc, 4),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "features": X.shape[1]
    }]).to_csv("models/metrics.csv", index=False)
    
    # Save feature names for serving
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n✅ Model saved! ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
