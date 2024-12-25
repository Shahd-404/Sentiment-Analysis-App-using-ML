import streamlit as st
import sklearn
import helper
import pickle
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
model = pickle.load(open("models/model.pkl", 'rb'))
vectorizer = pickle.load(open("models/vectorizer.pkl", 'rb'))

# Set app background color using markdown and HTML
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: yellow;
    }
    .stButton>button {
        background-color: black;
        color: white;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title for the app
st.title("Sentiment Analysis App")

# Input field for the user
text = st.text_input("Please enter your review:")

# Preprocess the text
token = helper.preprocessing_step(text)
vectorized_data = vectorizer.transform([token])

# Make a prediction
prediction = model.predict(vectorized_data)

# Display prediction when button is clicked
if st.button("Predict"):
    if prediction[0] == 0:
        st.text("Sentiment: Negative Review")
    else:
        st.text("Sentiment: Positive Review")
