import os
import sys
import pandas as pd
import requests
import io

dataset_dir = 'dataset'
train_csv_path = os.path.join(dataset_dir, 'train.csv')

os.makedirs(dataset_dir, exist_ok=True)

# Define file paths
kaggle_local_path = os.path.join(dataset_dir, 'kaggle_train.csv')
liar_local_path = os.path.join(dataset_dir, 'liar_train.tsv')

print("="*60)
print("Downloading Datasets...")
print("="*60)

# 1. Download Kaggle Dataset (via Public GitHub Raw Link as primary or fallback)
kaggle_url = "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv"
print(f"Downloading Kaggle Fake News dataset from public raw URL...")
try:
    r = requests.get(kaggle_url, timeout=30)
    r.raise_for_status()
    with open(kaggle_local_path, 'wb') as f:
        f.write(r.content)
    print("[+] Kaggle dataset downloaded successfully.")
except Exception as e:
    print(f"[-] Failed to download Kaggle dataset from GitHub: {e}")
    # Try Kaggle API as fallback
    print("Trying Kaggle API as fallback...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.competition_download_files('fake-news', path=dataset_dir)
        import zipfile
        zip_path = os.path.join(dataset_dir, 'fake-news.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(zip_path)
            # rename to kaggle_train.csv
            os.rename(os.path.join(dataset_dir, 'train.csv'), kaggle_local_path)
            print("[+] Kaggle dataset downloaded via Kaggle API.")
    except Exception as e_api:
        print(f"[-] Kaggle API fallback failed: {e_api}")

# 2. Download LIAR Dataset
liar_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
print(f"Downloading LIAR dataset from raw URL...")
try:
    r = requests.get(liar_url, timeout=30)
    r.raise_for_status()
    with open(liar_local_path, 'wb') as f:
        f.write(r.content)
    print("[+] LIAR dataset downloaded successfully.")
except Exception as e:
    print(f"[-] Failed to download LIAR dataset: {e}")

# 3. Combine Datasets
if os.path.exists(kaggle_local_path) and os.path.exists(liar_local_path):
    print("Combining datasets to mitigate single-source bias...")
    # Load Kaggle
    df_kaggle = pd.read_csv(kaggle_local_path)
    df_kaggle = df_kaggle.dropna(subset=['text'])
    
    # Robust column mapping for Kaggle/joolsa dataset
    cols = df_kaggle.columns
    df_kaggle_clean = pd.DataFrame()
    df_kaggle_clean['text'] = df_kaggle['text']
    
    # Map label
    if 'label' in cols:
        # Check if labels are text FAKE/REAL by inspect unique values
        labels_unique = [str(x).upper() for x in df_kaggle['label'].dropna().unique()]
        if 'FAKE' in labels_unique or 'REAL' in labels_unique:
            label_map = {'FAKE': 1, 'REAL': 0, 'fake': 1, 'real': 0}
            df_kaggle_clean['label'] = df_kaggle['label'].astype(str).str.upper().map(label_map).fillna(0).astype(int)
        else:
            df_kaggle_clean['label'] = pd.to_numeric(df_kaggle['label'], errors='coerce').fillna(0).astype(int)
    else:
        df_kaggle_clean['label'] = 0
        
    df_kaggle_clean['title'] = df_kaggle['title'] if 'title' in cols else ""
    df_kaggle_clean['author'] = df_kaggle['author'] if 'author' in cols else ""
    df_kaggle_clean['source'] = 'Kaggle'
    df_kaggle = df_kaggle_clean
    
    # Load LIAR
    # LIAR TSV structure has no header, we define names
    liar_cols = [
        'id', 'label_raw', 'statement', 'subjects', 'speaker', 
        'speaker_job', 'state', 'party', 'barely_true', 'false', 
        'half_true', 'mostly_true', 'pants_on_fire', 'context'
    ]
    df_liar = pd.read_csv(liar_local_path, sep='\t', header=None, names=liar_cols, quoting=3) # quoting=3 is csv.QUOTE_NONE
    
    # Map LIAR veracity labels to binary:
    # pants-on-fire, false, barely-true -> 1 (Fake)
    # half-true, mostly-true, true -> 0 (Real)
    liar_mapping = {
        'pants-fire': 1,
        'false': 1,
        'barely-true': 1,
        'half-true': 0,
        'mostly-true': 0,
        'true': 0
    }
    df_liar['label'] = df_liar['label_raw'].map(liar_mapping)
    # Drop rows where label couldn't be mapped
    df_liar = df_liar.dropna(subset=['label', 'statement'])
    df_liar['label'] = df_liar['label'].astype(int)
    
    # Standardize columns
    df_liar_clean = pd.DataFrame()
    df_liar_clean['text'] = df_liar['statement']
    df_liar_clean['label'] = df_liar['label']
    df_liar_clean['title'] = df_liar['subjects'] # use subjects as title
    df_liar_clean['author'] = df_liar['speaker'] # use speaker as author
    df_liar_clean['source'] = 'LIAR'
    
    # Concatenate
    df_combined = pd.concat([df_kaggle, df_liar_clean], ignore_index=True)
    df_combined.to_csv(train_csv_path, index=False)
    print(f"[+] Datasets combined successfully. Saved to {train_csv_path}")
    print(f"    Kaggle samples: {len(df_kaggle)}")
    print(f"    LIAR samples: {len(df_liar_clean)}")
    print(f"    Combined samples: {len(df_combined)}")
    
    # Clean up temp files
    os.remove(kaggle_local_path)
    os.remove(liar_local_path)
elif os.path.exists(kaggle_local_path):
    print("[-] LIAR dataset not available. Using Kaggle dataset only.")
    os.rename(kaggle_local_path, train_csv_path)
elif os.path.exists(liar_local_path):
    print("[-] Kaggle dataset not available. Using LIAR dataset only.")
    # Map and rename
    liar_cols = [
        'id', 'label_raw', 'statement', 'subjects', 'speaker', 
        'speaker_job', 'state', 'party', 'barely_true', 'false', 
        'half_true', 'mostly_true', 'pants_on_fire', 'context'
    ]
    df_liar = pd.read_csv(liar_local_path, sep='\t', header=None, names=liar_cols, quoting=3)
    liar_mapping = {
        'pants-fire': 1, 'false': 1, 'barely-true': 1,
        'half-true': 0, 'mostly-true': 0, 'true': 0
    }
    df_liar['label'] = df_liar['label_raw'].map(liar_mapping)
    df_liar = df_liar.dropna(subset=['label', 'statement'])
    df_liar['label'] = df_liar['label'].astype(int)
    
    df_liar_clean = pd.DataFrame()
    df_liar_clean['text'] = df_liar['statement']
    df_liar_clean['label'] = df_liar['label']
    df_liar_clean['title'] = df_liar['subjects']
    df_liar_clean['author'] = df_liar['speaker']
    df_liar_clean['source'] = 'LIAR'
    df_liar_clean.to_csv(train_csv_path, index=False)
    os.remove(liar_local_path)
    print(f"[+] LIAR dataset processed and saved to {train_csv_path}")
else:
    print("[-] Failed to download any dataset. Please place train.csv in dataset/ manually.")
    sys.exit(1)
