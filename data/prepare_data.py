import pandas as pd
import requests
import gzip
import io
from Bio import SeqIO
from collections import Counter

def download_clinvar():
    """Download ClinVar variant summary — real clinical data"""
    print("📥 Downloading ClinVar dataset...")
    
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    response = requests.get(url, stream=True)
    
    with gzip.open(io.BytesIO(response.content), 'rt') as f:
        df = pd.read_csv(f, sep='\t', low_memory=False)
    
    print(f"✅ Downloaded {len(df):,} variants")
    return df

def prepare_dataset(df, max_samples=5000):
    """
    Filter to SNVs (single nucleotide variants) only
    Label: 1 = Pathogenic, 0 = Benign
    """
    print("🔬 Preparing dataset...")
    
    # Keep only SNVs with clear pathogenicity labels
    snvs = df[
        (df['Type'] == 'single nucleotide variant') &
        (df['ClinicalSignificance'].isin(['Pathogenic', 'Benign'])) &
        (df['ReferenceAlleleVCF'].str.len() == 1) &
        (df['AlternateAlleleVCF'].str.len() == 1)
    ].copy()
    
    # Create label
    snvs['label'] = (snvs['ClinicalSignificance'] == 'Pathogenic').astype(int)
    
    # Create DNA sequence context (ref + alt + gene info as string)
    snvs['sequence'] = snvs.apply(
        lambda r: f"{'A' * 50}{r['ReferenceAlleleVCF']}{r['AlternateAlleleVCF']}{'T' * 50}", 
        axis=1
    )
    
    # Balance classes
    pathogenic = snvs[snvs['label'] == 1].head(max_samples // 2)
    benign     = snvs[snvs['label'] == 0].head(max_samples // 2)
    balanced   = pd.concat([pathogenic, benign]).sample(frac=1, random_state=42)
    
    print(f"✅ Dataset ready:")
    print(f"   Pathogenic: {len(pathogenic):,}")
    print(f"   Benign:     {len(benign):,}")
    print(f"   Total:      {len(balanced):,}")
    
    return balanced[['sequence', 'label', 'GeneSymbol', 
                     'ClinicalSignificance', 'Chromosome']].reset_index(drop=True)

if __name__ == "__main__":
    df  = download_clinvar()
    data = prepare_dataset(df)
    data.to_csv("data/clinvar_prepared.csv", index=False)
    print("💾 Saved to data/clinvar_prepared.csv")
    print(data.head())
