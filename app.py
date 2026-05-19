import streamlit as st
import pickle
import re
import string

# ── Load stopwords ──────────────────────────────────────────────────────────
try:
    with open('/root/nltk_data/corpora/stopwords/english', 'r') as f:
        stop_words = set(f.read().splitlines())
except:
    stop_words = {'i','me','my','we','our','you','your','he','him','his','she',
                  'her','it','its','they','them','what','this','that','these',
                  'those','am','is','are','was','were','be','been','have','has',
                  'had','do','does','did','a','an','the','and','but','if','or',
                  'as','of','at','by','for','with','in','out','on','to','from',
                  'up','not','no','so','can','will','just','now','also','only'}

# ── Simple stemmer ───────────────────────────────────────────────────────────
def simple_stem(word):
    suffixes = ['ing', 'tion', 'ed', 'er', 'ly', 'es', 's']
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) > 2:
            return word[:-len(suf)]
    return word

# ── Text transform (same as training) ────────────────────────────────────────
def transform_text(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)

# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextArea textarea {
        background-color: #1e1e2e;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        font-size: 15px;
    }
    .result-spam {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
    }
    .result-ham {
        background: linear-gradient(135deg, #00cc66, #007a3d);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,204,102,0.3);
    }
    .info-box {
        background-color: #1e1e2e;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 4px solid #4f8ef7;
        margin: 8px 0;
        color: #ccc;
        font-size: 14px;
    }
    .metric-box {
        background-color: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📧 Email / SMS Spam Classifier")
st.markdown("**GTU 8th Sem Project — Salunkhe Hitesh Popat (211310132023)**")
st.markdown("---")

# ── Model info ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box"><h4>🎯 Accuracy</h4><h2 style="color:#4f8ef7">97.29%</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><h4>✅ Precision</h4><h2 style="color:#00cc66">100%</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><h4>🤖 Model</h4><h2 style="color:#f4c430">Naive Bayes</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("### ✉️ Enter Email / SMS Text:")
input_text = st.text_area(
    label="Enter message", 
    placeholder="Type or paste your email/SMS message here...",
    height=160,  
    key="input_box"
)

# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_btn:
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            transformed = transform_text(input_text)
            # Vectorize
            vectorized = vectorizer.transform([transformed]).toarray()
            # Predict
            result = model.predict(vectorized)[0]
            # Probability
            prob = model.predict_proba(vectorized)[0]
            spam_prob = round(prob[1] * 100, 2)
            ham_prob  = round(prob[0] * 100, 2)

        st.markdown("### 📊 Result:")

        if result == 1:
            st.markdown('<div class="result-spam">🚨 SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">✅ NOT SPAM</div>', unsafe_allow_html=True)

        # Confidence
        st.markdown("#### 🔢 Confidence Scores:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🚨 Spam Probability", f"{spam_prob}%")
        with col_b:
            st.metric("✅ Ham Probability", f"{ham_prob}%")

        # Show preprocessing steps
        with st.expander("🔬 See Preprocessing Steps"):
            st.markdown(f"**Original Text:**\n```\n{input_text[:300]}\n```")
            st.markdown(f"**After Transform:**\n```\n{transformed[:300]}\n```")
            st.markdown(f"**Word Count (original):** {len(input_text.split())}")
            st.markdown(f"**Word Count (after preprocessing):** {len(transformed.split())}")

st.markdown("---")

# ── Try examples ──────────────────────────────────────────────────────────────
st.markdown("### 🧪 Try Example Messages:")

col_x, col_y = st.columns(2)

spam_examples = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Click here to claim.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121.",
    "Congratulations! You have won a FREE iPhone 15. Call now to claim your prize!",
    "URGENT: Your account has been suspended. Click link to verify: http://fake.com",
]

ham_examples = [
    "Hey, are you coming to college tomorrow? Let me know.",
    "I will be late for the meeting. Please start without me.",
    "Can you send me the notes from today's lecture please?",
    "Mom said dinner is ready. Come home soon.",
]

with col_x:
    st.markdown("**🚨 Spam Examples:**")
    for i, ex in enumerate(spam_examples):
        if st.button(f"Spam {i+1}", key=f"spam_{i}", use_container_width=True):
            st.session_state['input'] = ex
            st.rerun()

with col_y:
    st.markdown("**✅ Ham Examples:**")
    for i, ex in enumerate(ham_examples):
        if st.button(f"Ham {i+1}", key=f"ham_{i}", use_container_width=True):
            st.session_state['input'] = ex
            st.rerun()

st.markdown("---")

# ── Project info ──────────────────────────────────────────────────────────────
with st.expander("📚 About This Project"):
    st.markdown("""
    **Email Spam Prediction** — GTU 8th Semester Internship Project

    | Detail | Value |
    |---|---|
    | Student | Salunkhe Hitesh Popat |
    | Enrollment | 211310132023 |
    | Institute | Adani Institute of Infrastructure Engineering |
    | Guide | Dr. Ajay Kumar Vyas |
    | Internship | Project Tree, Surat |

    **Pipeline:**
    1. 📥 Dataset: spam.csv (5572 rows → 5169 after cleaning)
    2. 🧹 Preprocessing: Lowercase → Tokenize → Remove Stopwords → Stem
    3. 🔢 Feature Extraction: TF-IDF (max_features=3000)
    4. 🤖 Model: Multinomial Naive Bayes
    5. 📊 Results: 97.29% Accuracy, 100% Precision
    6. 🌐 Deployment: Streamlit Web App
    """)

st.markdown("<center><small>Built with ❤️ using Python + Streamlit + Scikit-learn</small></center>", unsafe_allow_html=True)
