import streamlit as st
import pandas as pd
import pdfplumber
import docx2txt
import joblib
import base64
import sqlite3
import bcrypt
from sklearn.feature_extraction.text import TfidfVectorizer

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    data = c.fetchone()
    conn.close()
    if data and bcrypt.checkpw(password.encode(), data[0]):
        return True
    return False

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please ensure 'best_model.pkl' and 'vectorizer.pkl' are available.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, vectorizer = load_model()
if model is None or vectorizer is None:
    st.stop()

def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/gif;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: black;
        }}
        .stApp * {{
            color: black !important;
        }}
        .stSidebar, .stSidebar * {{
            color: white !important;
        }}
        .stSelectbox label {{
            color: white !important;
        }}
        .stSelectbox div[data-baseweb="select"] * {{
            color: white !important;
        }}
        .stSelectbox div[data-baseweb="select"] ul {{
            background-color: #333 !important;
        }}
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image '{image_path}' not found.")

set_background("rs.gif")

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(uploaded_file)
        else:
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def predict_job_role(resume_text):
    resume_vector = vectorizer.transform([resume_text])
    return model.predict(resume_vector)[0]

def main():
    st.title("Resume Job Role Predictor 🔍")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if st.session_state.logged_in:
        st.success(f"Welcome {st.session_state.username} 👋")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        uploaded_file = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded_file:
            with st.spinner("Processing resume..."):
                resume_text = extract_text_from_file(uploaded_file)
                if resume_text:
                    st.subheader("Extracted Resume Text:")
                    st.text_area("Resume", resume_text, height=250)
                    with st.spinner("Predicting job role..."):
                        predicted_role = predict_job_role(resume_text)
                    st.success(f"Predicted Job Role: {predicted_role}")
                else:
                    st.error("Unable to process the uploaded file.")
    else:
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Choose Action", menu)
        if choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Login failed.")
        elif choice == "Sign Up":
            st.subheader("Sign Up")
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type='password')
            if st.button("Create Account"):
                if new_user and new_pass:
                    try:
                        add_user(new_user, new_pass)
                        st.success("User registered. Go to Login.")
                    except:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill both fields.")

if __name__ == "__main__":
    init_db()
    main()