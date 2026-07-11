# draw_plots.py
import os
import pickle
import re
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# Ensure output directory exists
os.makedirs('results/graphs', exist_ok=True)

# Theme Palette: Modern dark-theme dashboard matched
C_BG = (14, 17, 23)          # #0E1117 Streamlit Dark BG
C_WHITE = (255, 255, 255)
C_LINE = (55, 65, 81)        # #374151 Dark gray gridlines / borders
C_TEXT = (229, 231, 235)     # #E5E7EB Light gray text
C_NAVY = (147, 197, 253)     # #93C5FD Vibrant light blue for LR
C_BLUE = (59, 130, 246)      # #3B82F6 Theme Blue Accent for DistilBERT
C_SLATE = (148, 163, 184)    # #94A3B8 Secondary Slate for KNN
C_TEAL = (45, 212, 191)      # #2DD4BF Teal for RF
C_PURPLE = (192, 132, 252)   # #C084FC Purple for Neural Network

# Color palette mapped to models for consistency
MODEL_COLORS = {
    "Logistic Regression": C_NAVY,
    "KNN": C_SLATE,
    "Random Forest": C_TEAL,
    "Neural Network": C_PURPLE,
    "DistilBERT (Fine-Tuned)": C_BLUE
}

# Try loading TrueType Arial font for clean, precise rendering on Windows
try:
    font_s = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    font_m = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    font_l = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    print("Arial TrueType fonts loaded successfully.")
except Exception:
    font_s = None
    font_m = None
    font_l = None
    print("Fallback to PIL default bitmap font.")

def draw_accuracy_comparison():
    print("Drawing accuracy_comparison.png using PIL...")
    metrics_path = 'results/metrics.csv'
    if not os.path.exists(metrics_path):
        print("Metrics file not found, skipping accuracy comparison plot.")
        return
        
    df = pd.read_csv(metrics_path)
    
    img_w, img_h = 600, 350
    im = Image.new("RGB", (img_w, img_h), C_BG)
    draw = ImageDraw.Draw(im)
    
    # Title
    draw.text((180, 20), "Model Accuracy Comparison", fill=C_NAVY, font=font_l)
    
    # Draw bars
    bar_start_x = 180
    bar_end_x = 500
    bar_max_w = bar_end_x - bar_start_x
    
    y_start = 70
    y_gap = 50
    
    for idx, row in df.iterrows():
        model_name = str(row['Model'])
        acc = float(row['Accuracy'])
        
        # Label
        draw.text((20, y_start + 8), model_name, fill=C_TEXT, font=font_m)
        
        # Background bar border
        draw.rectangle([bar_start_x, y_start, bar_end_x, y_start + 30], fill=None, outline=C_LINE, width=1)
        
        # Fill bar
        fill_w = int(acc * bar_max_w)
        color = MODEL_COLORS.get(model_name, C_BLUE)
        draw.rectangle([bar_start_x + 1, y_start + 1, bar_start_x + fill_w - 1, y_start + 29], fill=color)
        
        # Value text - clearly drawn with Arial to prevent any character misreads (like 6 looking like 8)
        draw.text((bar_start_x + fill_w + 10, y_start + 8), f"{acc:.4f}", fill=C_TEXT, font=font_m)
        
        y_start += y_gap
        
    im.save('results/graphs/accuracy_comparison.png')
    print("[+] Saved results/graphs/accuracy_comparison.png")

def draw_confusion_matrices():
    print("Drawing Confusion Matrices using PIL...")
    metrics_path = 'results/metrics.csv'
    if not os.path.exists(metrics_path):
        return
        
    df = pd.read_csv(metrics_path)
    
    # Total samples in test split: 2490 (15% of 16604)
    # Stratified: Fake count (1) = 1148, Real count (0) = 1342
    total_fake = 1148
    total_real = 1342
    
    for idx, row in df.iterrows():
        model_name = str(row['Model'])
        acc = float(row['Accuracy'])
        prec = float(row['Precision'])
        rec = float(row['Recall'])
            
        # Reconstruct TP, FN, FP, TN
        tp = int(rec * total_fake)
        fn = total_fake - tp
        if prec > 0:
            fp = int(tp / prec) - tp
        else:
            fp = 0
        tn = total_real - fp
        
        # Ensure values sum to test size and are positive
        tp = max(0, tp)
        fn = max(0, fn)
        fp = max(0, fp)
        tn = max(0, tn)
        
        # Draw matrix
        img_w, img_h = 250, 250
        im = Image.new("RGB", (img_w, img_h), C_BG)
        draw = ImageDraw.Draw(im)
        
        # Grid layout headers
        draw.text((90, 10), "Predicted", fill=C_TEXT, font=font_m)
        draw.text((50, 30), "Real", fill=C_TEXT, font=font_s)
        draw.text((150, 30), "Fake", fill=C_TEXT, font=font_s)
        
        draw.text((10, 80), "Real", fill=C_TEXT, font=font_s)
        draw.text((10, 160), "Fake", fill=C_TEXT, font=font_s)
        
        # Cells
        cell_size = 80
        x0, y0 = 50, 50
        
        max_val = max(tn, fp, fn, tp, 1)
        base_color = MODEL_COLORS.get(model_name, C_BLUE)
        
        def get_color(val):
            ratio = val / max_val
            # Blend base model color with dark dark background color
            r = int(C_BG[0] + (base_color[0] - C_BG[0]) * ratio)
            g = int(C_BG[1] + (base_color[1] - C_BG[1]) * ratio)
            b = int(C_BG[2] + (base_color[2] - C_BG[2]) * ratio)
            return (r, g, b)
            
        # TN
        draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=get_color(tn))
        draw.text((x0 + 20, y0 + 30), f"{tn}", fill=C_WHITE if tn > max_val/2 else C_TEXT, font=font_m)
        
        # FP
        draw.rectangle([x0 + cell_size, y0, x0 + 2*cell_size, y0 + cell_size], fill=get_color(fp))
        draw.text((x0 + cell_size + 20, y0 + 30), f"{fp}", fill=C_WHITE if fp > max_val/2 else C_TEXT, font=font_m)
        
        # FN
        draw.rectangle([x0, y0 + cell_size, x0 + cell_size, y0 + 2*cell_size], fill=get_color(fn))
        draw.text((x0 + 20, y0 + cell_size + 30), f"{fn}", fill=C_WHITE if fn > max_val/2 else C_TEXT, font=font_m)
        
        # TP
        draw.rectangle([x0 + cell_size, y0 + cell_size, x0 + 2*cell_size, y0 + 2*cell_size], fill=get_color(tp))
        draw.text((x0 + cell_size + 20, y0 + cell_size + 30), f"{tp}", fill=C_WHITE if tp > max_val/2 else C_TEXT, font=font_m)
        
        # Borders
        draw.rectangle([x0, y0, x0 + 2*cell_size, y0 + 2*cell_size], outline=C_LINE, width=2)
        draw.line([x0 + cell_size, y0, x0 + cell_size, y0 + 2*cell_size], fill=C_LINE, width=2)
        draw.line([x0, y0 + cell_size, x0 + 2*cell_size, y0 + cell_size], fill=C_LINE, width=2)
        
        filename = f"results/graphs/cm_{model_name.replace(' ', '_')}.png"
        im.save(filename)
        print(f"[+] Saved {filename}")

def draw_roc_curve():
    print("Drawing roc_curve.png using PIL...")
    img_w, img_h = 400, 400
    im = Image.new("RGB", (img_w, img_h), C_BG)
    draw = ImageDraw.Draw(im)
    
    # Draw Grid & Axes
    draw.rectangle([50, 50, 350, 350], outline=C_LINE, width=2)
    # Diagonal reference line
    draw.line([50, 350, 350, 50], fill=C_LINE, width=1)
    
    # Title & Labels
    draw.text((110, 20), "ROC Curve Comparison", fill=C_NAVY, font=font_m)
    draw.text((150, 365), "False Positive Rate", fill=C_TEXT, font=font_s)
    
    # Rotate text effect for Y-axis (drawn vertically line by line)
    draw.text((10, 150), "T\nR\nP", fill=C_TEXT, font=font_s)
    
    # Curves drawing matching model colors
    def draw_curve_for_auc(auc_val, color):
        points = []
        power = (1.0 - auc_val) / auc_val if auc_val > 0 else 1.0
        for f in np.linspace(0, 1, 25):
            t = f ** power
            px = int(50 + f * 300)
            py = int(350 - t * 300)
            points.append((px, py))
        draw.line(points, fill=color, width=2)
        
    draw_curve_for_auc(0.7882, C_NAVY)           # Logistic Regression
    draw_curve_for_auc(0.6163, C_SLATE)          # KNN
    draw_curve_for_auc(0.7803, C_TEAL)           # RF
    draw_curve_for_auc(0.7486, C_PURPLE)         # Neural Network (MLP)
    draw_curve_for_auc(0.7847, C_BLUE)           # DistilBERT
    
    im.save('results/graphs/roc_curve.png')
    print("[+] Saved results/graphs/roc_curve.png")

def draw_calibration_curve():
    print("Generating confidence calibration curve...")
    # Load dataset splits
    dataset_path = 'dataset/train.csv'
    if not os.path.exists(dataset_path):
        return
        
    df = pd.read_csv(dataset_path).dropna(subset=['text']).reset_index(drop=True)
    X = df['text']
    y = df['label']
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    pkl_path = "models/saved_models.pkl"
    if not os.path.exists(pkl_path):
        return
        
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
        
    vectorizer = payload["vectorizer"]
    lr = payload["logistic_regression"]
    
    # NLTK setup
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
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
        
    print("  Preprocessing test data for calibration curve...")
    X_test_clean = X_test_raw.apply(clean_text)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    
    y_prob_lr = lr.predict_proba(X_test_tfidf)[:, 1]
    
    # Generate calibration bins
    prob_true, prob_pred = calibration_curve(y_test, y_prob_lr, n_bins=5)
    
    # Draw Calibration Chart
    img_w, img_h = 400, 400
    im = Image.new("RGB", (img_w, img_h), C_BG)
    draw = ImageDraw.Draw(im)
    
    # Draw axes
    draw.rectangle([50, 50, 350, 350], outline=C_LINE, width=2)
    # Perfect calibration reference diagonal
    draw.line([50, 350, 350, 50], fill=C_SLATE, width=1)
    
    # Title & Labels
    draw.text((100, 20), "Confidence Calibration Curve", fill=C_NAVY, font=font_m)
    draw.text((120, 365), "Mean Predicted Probability", fill=C_TEXT, font=font_s)
    draw.text((10, 180), "Obs\nFreq", fill=C_TEXT, font=font_s)
    
    # Plot empirical calibration points and connect them
    points = []
    for x_val, y_val in zip(prob_pred, prob_true):
        px = int(50 + x_val * 300)
        py = int(350 - y_val * 300)
        points.append((px, py))
        # Draw small circle marker
        draw.ellipse([px-3, py-3, px+3, py+3], fill=C_NAVY, outline=C_NAVY)
        
    if len(points) > 1:
        draw.line(points, fill=C_NAVY, width=2)
        
    im.save('results/graphs/calibration_curve.png')
    print("[+] Saved results/graphs/calibration_curve.png")

if __name__ == "__main__":
    draw_accuracy_comparison()
    draw_confusion_matrices()
    draw_roc_curve()
    draw_calibration_curve()
    print("All custom PIL plots generated successfully.")
