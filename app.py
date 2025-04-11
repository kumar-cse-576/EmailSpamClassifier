import streamlit as st
import pickle
import nltk
import string
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page configuration
st.set_page_config(
    page_title="Spam Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Pure Dark Theme with advanced styling
st.markdown("""
    <style>
    :root {
        --dark-1: #121212;
        --dark-2: #1E1E1E;
        --dark-3: #252525;
        --dark-4: #2E2E2E;
        --text: #FFFFFF;
        --text-secondary: #B0B0B0;
        --accent: #BB86FC;
        --accent-dark: #3700B3;
        --danger: #CF6679;
        --success: #03DAC6;
    }

    body {
        background-color: var(--dark-1);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .main {
       
    }

    .title {
     background-color: var(--dark-2);
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        max-width: 800px;
        margin: 0 auto;
        border: 1px solid var(--dark-4);
        font-size: 2.5rem;
        text-align: center;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }

    .description {
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    .result-container {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        background: var(--dark-3);
        border-left: 4px solid;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .result {
    color:white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    .spam {
        border-color: var(--danger);
    }

    .ham {
        border-color: var(--success);
    }

    .stTextArea textarea {
        background-color: var(--dark-3) !important;
        color: var(--text) !important;
        border: 1px solid var(--dark-4) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 1rem !important;
        min-height: 150px !important;
        transition: border 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(187, 134, 252, 0.2) !important;
    }

    .stTextArea label {
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 8px !important;
    }

    .stButton>button {
        background: var(--accent) !important;
        color: var(--dark-1) !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 1rem !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        margin-top: 10px !important;
    }

    .stButton>button:hover {
        background: var(--accent-dark) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(187, 134, 252, 0.25) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }

    .footer {
        text-align: center;
        margin-top: 2rem;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }

    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-container {
        animation: fadeIn 0.4s ease-out forwards;
    }
    </style>
""", unsafe_allow_html=True)

# App Layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🔍 Spam Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Detect spam messages with machine learning. Enter an email or SMS message to analyze.</div>',
    unsafe_allow_html=True)

input_sms = st.text_area("Message Input", placeholder="Paste your message here...")

if st.button("Analyze Message"):
    if not input_sms.strip():
        st.warning("Please enter a message to analyze")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vect_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vect_input)[0]

        # Show result with animation
        with st.spinner('Analyzing...'):
            import time

            time.sleep(0.8)  # Simulate processing time

            if result == 1:
                st.markdown("""
                    <div class="result-container spam">
                        <p class="result">⚠️ <span>Spam Detected</span></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-container ham">
                        <p class="result">✓ <span>Legitimate Message</span></p>
                    </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="footer">Machine Learning Model | Natural Language Processing</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)