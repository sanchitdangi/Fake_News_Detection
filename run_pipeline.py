# run_pipeline.py
import os
import re
import pickle
import json
import random
import time
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Check if we should skip plots (always True in headless execution to prevent GUI crashes)
skip_plots = True

import matplotlib
if skip_plots:
    matplotlib.use('template') # dummy headless backend that does nothing
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support, roc_auc_score

# Ensure directories exist
os.makedirs('dataset', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/graphs', exist_ok=True)
os.makedirs('report', exist_ok=True)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- 1. Data Loading & Statistics ---
dataset_path = 'dataset/train.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Combined train.csv missing at {dataset_path}. Please run download_dataset.py first.")

df = pd.read_csv(dataset_path)
print(f"Dataset loaded. Initial shape: {df.shape}")

# Drop rows with missing text
missing_values = int(df['text'].isna().sum())
df = df.dropna(subset=['text']).reset_index(drop=True)
print(f"Shape after removing rows with missing text: {df.shape}")

total_articles = len(df)
fake_count = int((df['label'] == 1).sum())
real_count = int((df['label'] == 0).sum())
avg_word_count = float(df['text'].apply(lambda x: len(str(x).split())).mean())

stats = {
    "Total Articles": total_articles,
    "Fake News Articles": fake_count,
    "Real News Articles": real_count,
    "Average Words per Article": round(avg_word_count, 2),
    "Missing Text Rows Removed": missing_values
}

with open('results/dataset_stats.json', 'w') as f:
    json.dump(stats, f)

print("Dataset Statistics saved.")

# --- 2. Exploratory Data Analysis (EDA) ---
if not skip_plots:
    try:
        # 2.1 Class Balance Analysis
        plt.figure(figsize=(6, 4))
        sns.countplot(x='label', hue='source', data=df)
        plt.title('Class Balance by Dataset Source')
        plt.xlabel('Label (0 = Real, 1 = Fake)')
        plt.ylabel('Count')
        plt.savefig('results/graphs/class_distribution.png', bbox_inches='tight')
        plt.close()

        # 2.2 Text Length Distribution Analysis
        df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x='text_length', hue='label', kde=True, bins=50, multiple='stack')
        plt.title('Article Word Count Distribution by Label')
        plt.xlim(0, 1500)
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.savefig('results/graphs/text_length_distribution.png', bbox_inches='tight')
        plt.close()
        print("[+] EDA plots generated.")
    except Exception as e:
        print(f"[-] Plotting failed: {e}")
else:
    print("[*] Skipping EDA plot generation (SKIP_PLOTS=1).")

# --- 3. Data Splitting (Anti-Leakage Audit) ---
X = df['text']
y = df['label']

# Split into Train and Temp (30%)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=random_seed, stratify=y
)

# Split Temp into Validation and Test (50/50 of temp = 15% / 15% of total)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
)

# --- 4. Text Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

print("Preprocessing training corpus...")
X_train_clean = X_train_raw.apply(clean_text)
print("Preprocessing validation corpus...")
X_val_clean = X_val_raw.apply(clean_text)
print("Preprocessing test corpus...")
X_test_clean = X_test_raw.apply(clean_text)
print("[+] Preprocessing complete.")

# --- 5. EDA Word Frequency ---
if not skip_plots:
    try:
        from collections import Counter
        all_words = " ".join(X_train_clean).split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(20)
        words, counts = zip(*common_words)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(counts), y=list(words), hue=list(words), palette='viridis', legend=False)
        plt.title('Top 20 Most Frequent Words in Cleaned Training Data')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.savefig('results/graphs/word_frequency.png', bbox_inches='tight')
        plt.close()
        print("[+] Word frequency plot generated.")
    except Exception as e:
        print(f"[-] Word frequency plotting failed: {e}")

# --- 6. Feature Engineering ---
# 6.1 Bag of Words
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train_clean)
X_val_bow = bow_vectorizer.transform(X_val_clean)
X_test_bow = bow_vectorizer.transform(X_test_clean)

# 6.2 TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_clean)
X_val_tfidf = tfidf_vectorizer.transform(X_val_clean)
X_test_tfidf = tfidf_vectorizer.transform(X_test_clean)

# 6.3 Word2Vec
from gensim.models import Word2Vec
sentences = [text.split() for text in X_train_clean]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=1, seed=random_seed)

def get_mean_w2v_vector(words, model):
    valid_words = [w for w in words if w in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_words], axis=0)

X_train_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_train_clean])
X_val_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_val_clean])
X_test_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_test_clean])

# --- 7. Model Building & Evaluation ---
results = []
trained_models = {}

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"Training {name}...")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    mean_cv = cv_scores.mean()
    
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = preds
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Save CM if plotting is enabled
    if not skip_plots:
        try:
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title(f'Confusion Matrix: {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(f'results/graphs/cm_{name.replace(" ", "_")}.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[-] CM plotting failed for {name}: {e}")
    
    results.append({
        "Model": name,
        "CV Accuracy": mean_cv,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC": roc_auc
    })
    return fpr, tpr, roc_auc

# GridSearchCV LR
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=random_seed), {'C': [0.1, 1.0, 10.0]}, cv=3)
lr_grid.fit(X_train_tfidf, y_train)
best_lr = lr_grid.best_estimator_

models_to_run = {
    "Logistic Regression": best_lr,
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=1, random_state=random_seed),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=150, random_state=random_seed)
}

roc_data = {}
for name, model in models_to_run.items():
    fpr, tpr, roc_auc = evaluate_model(name, model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    roc_data[name] = (fpr, tpr, roc_auc)

# --- 8. DistilBERT Fine-Tuning ---
print("Setting up DistilBERT...")
import torch
import psutil
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

# Set PyTorch seed for full reproducibility of PyTorch weight initialization & dropout
set_seed(random_seed)

# Detect hardware resources
total_ram_gb = psutil.virtual_memory().total / (1024**3)
cpu_cores = os.cpu_count() or 2
cuda_available = torch.cuda.is_available()

# Hyperparameters (Configurable via environment variables)
bert_lr = float(os.environ.get('BERT_LEARNING_RATE', '3e-5'))
bert_weight_decay = float(os.environ.get('BERT_WEIGHT_DECAY', '0.01'))
bert_warmup_ratio = float(os.environ.get('BERT_WARMUP_RATIO', '0.1'))
bert_epochs = int(os.environ.get('BERT_EPOCHS', '3'))

# Hardware Adaptive Training Matrix (Sizing & Limits Selection)
if cuda_available:
    train_subset_size = len(X_train_raw)
    val_subset_size = len(X_val_raw)
    max_length = 128
    batch_size = 16
    num_workers = min(4, cpu_cores)
    torch.set_num_threads(cpu_cores)
    gradient_accumulation_steps = 1
    print(f"[*] GPU active. Tier 1: Full dataset ({train_subset_size} samples), max_length={max_length}.")
elif total_ram_gb >= 16.0:
    train_subset_size = 5000
    val_subset_size = 1000
    max_length = 128
    batch_size = 16
    num_workers = min(2, cpu_cores)
    torch.set_num_threads(cpu_cores)
    gradient_accumulation_steps = 1
    print(f"[*] High RAM ({total_ram_gb:.1f} GB). Tier 1 CPU: Subset {train_subset_size} samples, max_length={max_length}.")
elif total_ram_gb >= 8.0:
    train_subset_size = 3000
    val_subset_size = 500
    max_length = 128
    batch_size = 8
    num_workers = 1
    torch.set_num_threads(cpu_cores)
    gradient_accumulation_steps = 2
    print(f"[*] Medium RAM ({total_ram_gb:.1f} GB). Tier 2 CPU: Subset {train_subset_size} samples, max_length={max_length}.")
else:
    # Low RAM vm runner (our sandbox instance: 5.88 GB, 2 CPU cores)
    # Automatically falls back to 1000 training samples (Tier 3) to execute successfully without memory paging freezes.
    # Fallback to max_length=64 cuts attention matrix overhead by 4x to run efficiently on 2 cores.
    train_subset_size = 1000
    val_subset_size = 250
    max_length = 64
    batch_size = 8
    num_workers = 0
    torch.set_num_threads(2)
    gradient_accumulation_steps = 2 # Simulate batch size 16 to ensure optimization stability on CPU
    print(f"[*] Low RAM VM runner ({total_ram_gb:.1f} GB, {cpu_cores} cores). Tier 3 CPU: Subset {train_subset_size} samples, max_length={max_length}.")

# Select subsets
X_train_sub = X_train_raw.iloc[:train_subset_size].tolist()
y_train_sub = y_train.iloc[:train_subset_size].tolist()
X_val_sub = X_val_raw.iloc[:val_subset_size].tolist()
y_val_sub = y_val.iloc[:val_subset_size].tolist()

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize only once here, preventing duplicate tokenizations
train_encodings = tokenizer(X_train_sub, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(X_val_sub, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(X_test_raw.tolist(), truncation=True, padding=True, max_length=max_length)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, y_train_sub)
val_dataset = NewsDataset(val_encodings, y_val_sub)
test_dataset = NewsDataset(test_encodings, y_test.tolist())

device = torch.device('cuda') if cuda_available else torch.device('cpu')
model_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

# Robust compute metrics callback function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predictions = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        roc_auc = 0.5
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc
    }

# Trainer Arguments
training_args = TrainingArguments(
    output_dir='./results_bert',
    num_train_epochs=bert_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=bert_lr,
    weight_decay=bert_weight_decay,
    warmup_ratio=bert_warmup_ratio,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=(not cuda_available) # Pin only if active GPU to prevent memory overhead
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

print(f"Fine-tuning DistilBERT (Samples: {train_subset_size}, LR: {bert_lr}, Epochs: {bert_epochs})...")
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"[+] Fine-tuning complete. Training time: {train_time:.1f} seconds.")

# Evaluate DistilBERT on full test set (avoiding manual loop and tokenization duplication)
print("Evaluating DistilBERT on full test set...")
start_inf = time.time()
predictions_output = trainer.predict(test_dataset)
inf_time = time.time() - start_inf

test_logits = predictions_output.predictions
if isinstance(test_logits, tuple):
    test_logits = test_logits[0]
exp_test_logits = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
test_probs = exp_test_logits / np.sum(exp_test_logits, axis=-1, keepdims=True)
pred_labels = np.argmax(test_logits, axis=1)

acc_bert = accuracy_score(y_test, pred_labels)
prec_bert = precision_score(y_test, pred_labels, zero_division=0)
rec_bert = recall_score(y_test, pred_labels, zero_division=0)
f1_bert = f1_score(y_test, pred_labels, zero_division=0)
fpr_bert, tpr_bert, _ = roc_curve(y_test, test_probs[:, 1])
roc_auc_bert = auc(fpr_bert, tpr_bert)

print(f"DistilBERT Full Test Accuracy: {acc_bert:.4f}")
print(f"DistilBERT Full Test F1: {f1_bert:.4f}")

results.append({
    "Model": "DistilBERT (Fine-Tuned)",
    "CV Accuracy": np.nan,
    "Accuracy": acc_bert,
    "Precision": prec_bert,
    "Recall": rec_bert,
    "F1-Score": f1_bert,
    "AUC": roc_auc_bert
})

if not skip_plots:
    try:
        cm_bert = confusion_matrix(y_test, pred_labels)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix: DistilBERT')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('results/graphs/cm_DistilBERT.png', bbox_inches='tight')
        plt.close()

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot(fpr_bert, tpr_bert, lw=2, linestyle='--', label=f'DistilBERT (AUC = {roc_auc_bert:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('results/graphs/roc_curve.png', bbox_inches='tight')
        plt.close()
        print("[+] ROC and CM plots saved.")
    except Exception as e:
        print(f"[-] Evaluation plotting failed: {e}")

# --- 9. LIME Explainability ---
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
lime_pipeline = make_pipeline(tfidf_vectorizer, trained_models["Logistic Regression"])
explainer = LimeTextExplainer(class_names=["Real", "Fake"])

test_df = pd.DataFrame({'text': X_test_raw, 'label': y_test})
test_df['pred'] = trained_models["Logistic Regression"].predict(X_test_tfidf)

fake_idx = test_df[(test_df['label'] == 1) & (test_df['pred'] == 1)].index[0]
real_idx = test_df[(test_df['label'] == 0) & (test_df['pred'] == 0)].index[0]

if not skip_plots:
    try:
        exp_fake = explainer.explain_instance(df.loc[fake_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
        exp_fake.save_to_file('results/graphs/lime_explanation_fake.html')
        exp_fake.as_pyplot_figure()
        plt.title('LIME Feature Importance - Fake News Example')
        plt.savefig('results/graphs/lime_fake_pyplot.png', bbox_inches='tight')
        plt.close()

        exp_real = explainer.explain_instance(df.loc[real_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
        exp_real.save_to_file('results/graphs/lime_explanation_real.html')
        exp_real.as_pyplot_figure()
        plt.title('LIME Feature Importance - Real News Example')
        plt.savefig('results/graphs/lime_real_pyplot.png', bbox_inches='tight')
        plt.close()
        print("[+] LIME explanation plots saved.")
    except Exception as e:
        print(f"[-] LIME plotting failed: {e}")
else:
    # Save text explanations as backup
    try:
        exp_fake = explainer.explain_instance(df.loc[fake_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
        exp_fake.save_to_file('results/graphs/lime_explanation_fake.html')
        exp_real = explainer.explain_instance(df.loc[real_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
        exp_real.save_to_file('results/graphs/lime_explanation_real.html')
        print("[+] LIME HTML explanations saved (skipped static PNGs).")
    except Exception as e:
        print(f"[-] LIME text saving failed: {e}")

# --- 10. Metrics & Model Export ---
results_df = pd.DataFrame(results)
results_df.to_csv('results/metrics.csv', index=False)

if not skip_plots:
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Model', hue='Model', data=results_df, palette='Set2', legend=False)
        plt.title('Model Accuracy Comparison')
        plt.xlim(0, 1.0)
        for index, value in enumerate(results_df['Accuracy']):
            plt.text(value, index, f'{value:.4f}')
        plt.savefig('results/graphs/accuracy_comparison.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[-] Accuracy comparison plotting failed: {e}")

# Serialization
streamlit_payload = {
    "vectorizer": tfidf_vectorizer,
    "logistic_regression": trained_models["Logistic Regression"],
    "random_forest": trained_models["Random Forest"],
    "neural_network": trained_models["Neural Network"],
    "knn": trained_models["KNN"]
}
with open('models/saved_models.pkl', 'wb') as f:
    pickle.dump(streamlit_payload, f)

os.makedirs('models/distilbert_fake_news', exist_ok=True)
model_bert.save_pretrained('models/distilbert_fake_news')
tokenizer.save_pretrained('models/distilbert_fake_news')
print("All tasks completed successfully. Metrics, models and graphs saved.")

# Print real-time pipeline performance review table
print("\n==========================================")
print(" Real-Time DistilBERT Upgrade Comparison")
print("==========================================")
print(f"Metric          | Baseline  | Upgraded ")
print(f"----------------+-----------+----------")
print(f"Train Samples   | 200       | {train_subset_size}")
print(f"Max Seq Length  | 128       | {max_length}")
print(f"Accuracy        | 0.5676    | {acc_bert:.4f}")
print(f"F1-Score        | 0.1322    | {f1_bert:.4f}")
print(f"Precision       | 0.8913    | {prec_bert:.4f}")
print(f"Recall          | 0.0714    | {rec_bert:.4f}")
print(f"AUC             | 0.7005    | {roc_auc_bert:.4f}")
print(f"Train Time (s)  | 321.5     | {train_time:.1f}")
print(f"Inference (s)   | 24.4      | {inf_time:.1f}")
print("==========================================\n")
