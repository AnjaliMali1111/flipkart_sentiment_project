# app.py

import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import base64

# -----------------------------
# Stopwords
# -----------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv('flipkart.csv', encoding='utf-8-sig')
except:
    df = pd.read_csv('flipkart.csv', encoding='latin1')

df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# -----------------------------
# Preprocess Text
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
all_reviews_text = " ".join(df['Cleaned_Review'].tolist())

# -----------------------------
# Prediction History
# -----------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = []

# -----------------------------
# Background + Logo
# -----------------------------
def add_bg_logo(bg_image, logo_image):
    def get_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    bg_b64 = get_base64(bg_image)
    logo_b64 = get_base64(logo_image)

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        section[data-testid="stSidebar"] {{
            background-image: url("data:image/png;base64,{logo_b64}");
            background-repeat: no-repeat;
            background-position: top center;
            background-size: 140px;
            padding-top: 160px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_logo("11.png", "logo.png")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Flipkart Sentiment Analysis App")

option = st.sidebar.radio("Go to:", ["Prediction", "WordCloud", "Analysis"])

# ---------- PREDICTION ----------
if option == "Prediction":
    st.subheader("Enter a review to predict sentiment")
    user_input = st.text_area("Your Text Here:")

    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text!")
        else:
            vector = vectorizer.transform([user_input])
            pred = model.predict(vector)[0]
            label = "Positive" if pred == 1 else "Negative"

            if pred == 1:
                st.success(f"Predicted Sentiment: {label}")
            else:
                st.error(f"Predicted Sentiment: {label}")

            st.session_state['history'].append((user_input, label))
            st.session_state['history'] = st.session_state['history'][-5:]

    if st.session_state['history']:
        st.subheader("Last 5 Predictions")
        st.table(st.session_state['history'])

# ---------- WORDCLOUD ----------
elif option == "WordCloud":
    st.subheader("WordCloud of Reviews by Sentiment")
    choice = st.selectbox("Select Sentiment", ["All", "Positive", "Negative"])

    if choice == "Positive":
        text = " ".join(df[df['Sentiment'] == 1]['Cleaned_Review'].tolist())
    elif choice == "Negative":
        text = " ".join(df[df['Sentiment'] == 0]['Cleaned_Review'].tolist())
    else:
        text = all_reviews_text

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ---------- ANALYSIS ----------
elif option == "Analysis":
    st.subheader("Sentiment Distribution")
    counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(['Positive','Negative'], [counts.get(1,0), counts.get(0,0)], color=['green','red'])
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Positive vs Negative Reviews")
    st.pyplot(fig)
