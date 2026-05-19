# 📧 Email Spam Prediction — Run Guide
## Salunkhe Hitesh Popat | 211310132023 | AIIE | GTU

---

## ✅ FILES IN THIS FOLDER
- app.py              → Main Streamlit web app
- model.pkl           → Trained Naive Bayes model
- vectorizer.pkl      → Trained TF-IDF vectorizer
- spam.csv            → Dataset (spam.csv)
- spam_ham_dataset.csv→ Dataset (ham)
- requirements.txt    → Python packages needed
- nltk_data/          → Stopwords (no download needed)

---

## 🚀 METHOD 1 — Run on YOUR Laptop (Recommended for Viva)

### Step 1: Install Python
- Download Python 3.9+ from https://python.org
- During install, CHECK "Add Python to PATH"

### Step 2: Open Command Prompt / Terminal
- Windows: Press Win+R → type cmd → Enter
- Mac/Linux: Open Terminal

### Step 3: Go to project folder
```
cd Desktop\spam_project
```

### Step 4: Install packages
```
pip install streamlit scikit-learn pandas numpy
```

### Step 5: Run the app
```
streamlit run app.py
```

### Step 6: Browser opens automatically at:
```
http://localhost:8501
```

---

## 🌐 METHOD 2 — Run on Streamlit Cloud (Show Online - FREE)

### Step 1: Upload to GitHub
1. Go to https://github.com → Sign in / Sign up (free)
2. Click "New Repository" → Name it: spam-classifier
3. Upload ALL files from this folder
4. Click "Commit changes"

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo → Branch: main → File: app.py
5. Click "Deploy!"
6. Wait 2-3 minutes → You get a PUBLIC link like:
   https://yourname-spam-classifier.streamlit.app

### Step 3: Show examiner this link — works from any device!

---

## ☁️ METHOD 3 — Google Colab (No install needed)

1. Go to https://colab.research.google.com
2. Upload app.py, model.pkl, vectorizer.pkl, nltk_data folder
3. Run this code in a cell:

```python
!pip install streamlit pyngrok -q

# Copy nltk data
import shutil, os
os.makedirs('/root/nltk_data/corpora/stopwords', exist_ok=True)
shutil.copy('nltk_data/corpora/stopwords/english', 
            '/root/nltk_data/corpora/stopwords/english')

# Start with ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("PUBLIC URL:", public_url)

!streamlit run app.py --server.port 8501 &
```

4. Click the public URL — share it with examiner!

---

## 📱 WHAT THE APP SHOWS

- Enter any email/SMS text
- Click "Predict" button
- Shows: SPAM (red) or NOT SPAM (green)
- Shows confidence percentage
- Shows preprocessing steps
- 8 example messages to try (4 spam + 4 ham)
- Project info and model metrics

---

## 🎯 MODEL PERFORMANCE
- Algorithm: Multinomial Naive Bayes
- Accuracy: 97.29%
- Precision: 100% (Zero false positives!)
- Dataset: 5169 emails after cleaning
- Features: TF-IDF with 3000 features

---

## ❓ IF SOMETHING GOES WRONG

Error: "streamlit not found"
→ Run: pip install streamlit

Error: "No module named sklearn"  
→ Run: pip install scikit-learn

Error: "Port already in use"
→ Run: streamlit run app.py --server.port 8502

App opens but model error:
→ Make sure model.pkl and vectorizer.pkl are in SAME folder as app.py
