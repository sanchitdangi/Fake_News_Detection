import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title and Overview
cells.append(nbf.v4.new_markdown_cell("""# Fake News Detection using Text Classification
This notebook implements an advanced Machine Learning pipeline to classify news articles as real or fake. It preserves baseline TF-IDF and traditional classifiers while introducing dataset improvements (combining Kaggle and LIAR datasets), proper validation splits, hyperparameter tuning, a fine-tuned DistilBERT transformer model, and LIME explainability.

## Pipeline Architecture:
1. **Reproducibility**: Set seeds for all libraries.
2. **Dataset Loading & Upgrades**: Combine Kaggle Fake News dataset and Politifact LIAR dataset to mitigate single-source bias.
3. **Exploratory Data Analysis (EDA)**: Sanity checks, class distribution, source breakdown, text length analysis, and word frequencies.
4. **Data Splitting**: Proper Train/Validation/Test split to avoid data leakage.
5. **Preprocessing**: Tokenization, stopword removal, lowercasing, punctuation cleaning, and WordNet lemmatization.
6. **Feature Extraction**: Bag of Words, TF-IDF (fit *only* on training data), and Word2Vec sentence embeddings.
7. **Model Building & Baseline Evaluation**: traditional ML models (KNN, Logistic Regression, Random Forest, MLPClassifier) with 5-Fold Cross-Validation and GridSearchCV tuning.
8. **Advanced Transformer Upgrade**: DistilBERT sequence classification fine-tuning.
9. **Explainable AI (XAI)**: LIME text explainer on the TF-IDF + Logistic Regression model.
10. **Model Comparison & Export**: Comparison metrics, plots, and serialization for the Streamlit deployment.
"""))

# Cell 1: Reproducibility & Environment Setup
cells.append(nbf.v4.new_code_cell("""import os
import re
import pickle
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# Ensure directories exist
os.makedirs('../dataset', exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../results/graphs', exist_ok=True)
os.makedirs('../report', exist_ok=True)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
"""))

# Cell 2: Dataset Loading & Verification
cells.append(nbf.v4.new_markdown_cell("## 1. Dataset Loading & Statistics"))
cells.append(nbf.v4.new_code_cell("""dataset_path = '../dataset/train.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Combined train.csv missing at {dataset_path}. Please run download_dataset.py first.")

df = pd.read_csv(dataset_path)
print(f"Dataset loaded. Initial shape: {df.shape}")

# Basic sanity checks
print("Missing values:")
print(df.isna().sum())

# Drop rows with missing text
df = df.dropna(subset=['text']).reset_index(drop=True)
print(f"Shape after removing rows with missing text: {df.shape}")

# Inspect columns and data sources
print("\\nSource breakdown:")
print(df['source'].value_counts() if 'source' in df.columns else "No source column present")
"""))

# Cell 3: Exploratory Data Analysis (EDA)
cells.append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"))
cells.append(nbf.v4.new_code_cell("""# 2.1 Class Balance Analysis
plt.figure(figsize=(6, 4))
sns.countplot(x='label', hue='source', data=df)
plt.title('Class Balance by Dataset Source')
plt.xlabel('Label (0 = Real, 1 = Fake)')
plt.ylabel('Count')
plt.savefig('../results/graphs/class_distribution.png', bbox_inches='tight')
plt.show()

# 2.2 Text Length Distribution Analysis
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='text_length', hue='label', kde=True, bins=50, multiple='stack')
plt.title('Article Word Count Distribution by Label')
plt.xlim(0, 1500) # clip for readability
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.savefig('../results/graphs/text_length_distribution.png', bbox_inches='tight')
plt.show()

# Summary stats on text length
print(df.groupby('label')['text_length'].describe())
"""))

# Cell 4: Train/Validation/Test Split (Anti-Leakage Audit)
cells.append(nbf.v4.new_markdown_cell("## 3. Data Splitting\nWe perform a train/validation/test split (70%/15%/15%) *before* doing any text preprocessing or feature engineering to prevent any data leakage from test/validation sets into the training pipeline."))
cells.append(nbf.v4.new_code_cell("""X = df['text']
y = df['label']

# Split into Train and Temp (30%)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=random_seed, stratify=y
)

# Split Temp into Validation and Test (50/50 of temp = 15% / 15% of total)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
)

print(f"Training set:   {X_train_raw.shape[0]} samples")
print(f"Validation set: {X_val_raw.shape[0]} samples")
print(f"Testing set:    {X_test_raw.shape[0]} samples")
"""))

# Cell 5: Upgraded Text Preprocessing
cells.append(nbf.v4.new_markdown_cell("## 4. Text Preprocessing"))
cells.append(nbf.v4.new_code_cell("""stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters (punctuation, digits)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and perform WordNet Lemmatization
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

print("Preprocessing training corpus...")
X_train_clean = X_train_raw.apply(clean_text)
print("Preprocessing validation corpus...")
X_val_clean = X_val_raw.apply(clean_text)
print("Preprocessing test corpus...")
X_test_clean = X_test_raw.apply(clean_text)
print("[+] Preprocessing complete.")
"""))

# Cell 6: EDA - Word Frequency on Preprocessed Training Data
cells.append(nbf.v4.new_code_cell("""from collections import Counter
all_words = " ".join(X_train_clean).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), hue=list(words), palette='viridis', legend=False)
plt.title('Top 20 Most Frequent Words in Cleaned Training Data')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.savefig('../results/graphs/word_frequency.png', bbox_inches='tight')
plt.show()
"""))

# Cell 7: Feature Engineering (BoW, TF-IDF, Word2Vec)
cells.append(nbf.v4.new_markdown_cell("## 5. Feature Engineering\nTo prevent data leakage, vectorizers are fit **only** on the training set, then transformed on the validation and test sets."))
cells.append(nbf.v4.new_code_cell("""# 5.1 Bag of Words (CountVectorizer)
print("Fitting Bag of Words vectorizer...")
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train_clean)
X_val_bow = bow_vectorizer.transform(X_val_clean)
X_test_bow = bow_vectorizer.transform(X_test_clean)

# 5.2 TF-IDF (TfidfVectorizer)
print("Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_clean)
X_val_tfidf = tfidf_vectorizer.transform(X_val_clean)
X_test_tfidf = tfidf_vectorizer.transform(X_test_clean)

# 5.3 Word2Vec embeddings
from gensim.models import Word2Vec
print("Training Word2Vec model on training sentences...")
sentences = [text.split() for text in X_train_clean]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, seed=random_seed)

def get_mean_w2v_vector(words, model):
    valid_words = [w for w in words if w in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_words], axis=0)

print("Generating Word2Vec embeddings...")
X_train_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_train_clean])
X_val_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_val_clean])
X_test_w2v = np.array([get_mean_w2v_vector(text.split(), w2v_model) for text in X_test_clean])
print("[+] Feature extraction complete.")
"""))

# Cell 8: Model Comparison & Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("## 6. Model Training & Tuning\nWe define a consistent evaluation function, perform 5-Fold Cross-Validation, and carry out GridSearchCV for Logistic Regression tuning."))
cells.append(nbf.v4.new_code_cell("""# Consistent evaluation function
results = []
trained_models = {}

def evaluate_model(name, model, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    print(f"--- Evaluating {name} ---")
    
    # 5-Fold Cross Validation on Training
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    mean_cv = cv_scores.mean()
    print(f"5-Fold CV Mean Accuracy: {mean_cv:.4f}")
    
    # Fit on all training data
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Predict on test
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = preds
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print(classification_report(y_test, preds))
    
    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'../results/graphs/cm_{name.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()
    
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

# Hyperparameter Tuning on Logistic Regression
print("Running GridSearchCV for Logistic Regression...")
lr_param_grid = {'C': [0.1, 1.0, 10.0]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=random_seed), lr_param_grid, cv=3, scoring='accuracy')
lr_grid.fit(X_train_tfidf, y_train)
best_lr = lr_grid.best_estimator_
print(f"Best parameter for Logistic Regression: {lr_grid.best_params_}")

# Models to compare
models_to_run = {
    "Logistic Regression": best_lr,
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=random_seed), # limit depth for fast train
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=150, random_state=random_seed) # smaller network for fast train
}

# Run evaluation on TF-IDF features
plt.figure(figsize=(10, 8))
for name, model in models_to_run.items():
    fpr, tpr, roc_auc = evaluate_model(name, model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
"""))

# Cell 9: DistilBERT Transformer Upgrade
cells.append(nbf.v4.new_markdown_cell("## 7. Advanced Transformer Upgrade (DistilBERT)\nWe implement a fine-tuning pipeline for DistilBERT using Hugging Face's `transformers` library, utilizing dynamic hardware adaptation and early stopping."))
cells.append(nbf.v4.new_code_cell("""import torch
import psutil
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, auc

# Set PyTorch seed for full reproducibility
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
    train_subset_size = 1000
    val_subset_size = 250
    max_length = 64
    batch_size = 8
    num_workers = 0
    torch.set_num_threads(2)
    gradient_accumulation_steps = 2
    print(f"[*] Low RAM VM runner ({total_ram_gb:.1f} GB, {cpu_cores} cores). Tier 3 CPU: Subset {train_subset_size} samples, max_length={max_length}.")

# Select subsets
X_train_sub = X_train_raw.iloc[:train_subset_size].tolist()
y_train_sub = y_train.iloc[:train_subset_size].tolist()
X_val_sub = X_val_raw.iloc[:val_subset_size].tolist()
y_val_sub = y_val.iloc[:val_subset_size].tolist()

# Tokenizer
print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train_sub, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(X_val_sub, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(X_test_raw.tolist(), truncation=True, padding=True, max_length=max_length)

# Dataset class
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

# Load pre-trained model
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

# Training arguments
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
    dataloader_pin_memory=(not cuda_available)
)

# Trainer
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
"""))

# Cell 10: DistilBERT Evaluation
cells.append(nbf.v4.new_code_cell("""# Evaluate DistilBERT
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

# Metrics
acc_bert = accuracy_score(y_test, pred_labels)
prec_bert = precision_score(y_test, pred_labels, zero_division=0)
rec_bert = recall_score(y_test, pred_labels, zero_division=0)
f1_bert = f1_score(y_test, pred_labels, zero_division=0)

# ROC
fpr_bert, tpr_bert, _ = roc_curve(y_test, test_probs[:, 1])
roc_auc_bert = auc(fpr_bert, tpr_bert)

print(f"DistilBERT Full Test Accuracy: {acc_bert:.4f}")
print(f"DistilBERT Full Test F1: {f1_bert:.4f}")

# Append to results
results.append({
    "Model": "DistilBERT (Fine-Tuned)",
    "CV Accuracy": np.nan,
    "Accuracy": acc_bert,
    "Precision": prec_bert,
    "Recall": rec_bert,
    "F1-Score": f1_bert,
    "AUC": roc_auc_bert
})

# Save Confusion Matrix
cm_bert = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix: DistilBERT')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('../results/graphs/cm_DistilBERT.png', bbox_inches='tight')
plt.close()

# Plot DistilBERT ROC Curve
plt.figure(1)
plt.plot(fpr_bert, tpr_bert, lw=2, linestyle='--', label=f'DistilBERT (AUC = {roc_auc_bert:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('../results/graphs/roc_curve.png', bbox_inches='tight')
plt.show()
"""))

# Cell 11: Explainable AI using LIME
cells.append(nbf.v4.new_markdown_cell("## 8. Explainable AI (LIME)\nWe use LIME to construct local text explanations for individual predictions, showing word feature importances."))
cells.append(nbf.v4.new_code_cell("""from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# Create a classification pipeline for LIME: Vectorizer + Model
lime_pipeline = make_pipeline(tfidf_vectorizer, trained_models["Logistic Regression"])

explainer = LimeTextExplainer(class_names=["Real", "Fake"])

# Let's explain one true positive (Fake news correctly classified) and one true negative (Real news correctly classified)
# We find indices in test set
test_df = pd.DataFrame({'text': X_test_raw, 'label': y_test})
test_df['pred'] = trained_models["Logistic Regression"].predict(X_test_tfidf)

fake_idx = test_df[(test_df['label'] == 1) & (test_df['pred'] == 1)].index[0]
real_idx = test_df[(test_df['label'] == 0) & (test_df['pred'] == 0)].index[0]

print("Generating LIME explanation for Fake News Article...")
exp_fake = explainer.explain_instance(df.loc[fake_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
exp_fake.save_to_file('../results/graphs/lime_explanation_fake.html')
exp_fake.as_pyplot_figure()
plt.title('LIME Feature Importance - Fake News Example')
plt.savefig('../results/graphs/lime_fake_pyplot.png', bbox_inches='tight')
plt.show()

print("Generating LIME explanation for Real News Article...")
exp_real = explainer.explain_instance(df.loc[real_idx, 'text'], lime_pipeline.predict_proba, num_features=10)
exp_real.save_to_file('../results/graphs/lime_explanation_real.html')
exp_real.as_pyplot_figure()
plt.title('LIME Feature Importance - Real News Example')
plt.savefig('../results/graphs/lime_real_pyplot.png', bbox_inches='tight')
plt.show()
"""))

# Cell 12: Model Comparison & Metric Export
cells.append(nbf.v4.new_markdown_cell("## 9. Model Comparison & Metric Export"))
cells.append(nbf.v4.new_code_cell("""results_df = pd.DataFrame(results)
display(results_df)

# Save metrics CSV
results_df.to_csv('../results/metrics.csv', index=False)
print("Metrics saved to results/metrics.csv")

# Plot Accuracy Comparison Graph
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', hue='Model', data=results_df, palette='Set2', legend=False)
plt.title('Model Accuracy Comparison (Baselines vs DistilBERT)')
plt.xlim(0, 1.0)
for index, value in enumerate(results_df['Accuracy']):
    plt.text(value, index, f'{value:.4f}')
plt.savefig('../results/graphs/accuracy_comparison.png', bbox_inches='tight')
plt.show()
"""))

# Cell 13: Model Serialization
cells.append(nbf.v4.new_code_cell("""# Save traditional models & vectorizer for Streamlit
streamlit_payload = {
    "vectorizer": tfidf_vectorizer,
    "logistic_regression": trained_models["Logistic Regression"],
    "random_forest": trained_models["Random Forest"],
    "neural_network": trained_models["Neural Network"],
    "knn": trained_models["KNN"]
}

with open('../models/saved_models.pkl', 'wb') as f:
    pickle.dump(streamlit_payload, f)

# Save fine-tuned DistilBERT tokenizer and weights
os.makedirs('../models/distilbert_fake_news', exist_ok=True)
model_bert.save_pretrained('../models/distilbert_fake_news')
tokenizer.save_pretrained('../models/distilbert_fake_news')
print("All models and vectorizers successfully serialized to models/")
"""))

nb['cells'] = cells

with open('notebooks/Fake_News_Detection.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook rebuilt successfully.")
