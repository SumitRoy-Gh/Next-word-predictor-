import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(page_title="Next Word Predictor", page_icon="📝", layout="centered")

# Load the model and tokenizer
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('models/next_word_prediction_model.h5')
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_resources()

# Prediction function
def predict_next_word(model, tokenizer, text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Determine the required input length for the model
    # The model was trained with sequences of length max_sequence_len - 1
    # In the notebook, max_sequence_len was 14, so input shape was (None, 13)
    # However, let's get it dynamically
    required_input_length = model.input_shape[1]
    
    # If the token list is longer than the required input, truncate it from the left
    if len(token_list) > required_input_length:
        token_list = token_list[-required_input_length:]
    
    # Pad the sequence to the required input length
    padded_token_list = pad_sequences([token_list], maxlen=required_input_length, padding='pre')
    
    # Make prediction
    predicted = model.predict(padded_token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    .prediction-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

# UI Layout
st.title("✨ Next Word Predictor")
st.markdown("Enter a phrase below and let the model predict the most likely next word.")

if model and tokenizer:
    input_text = st.text_input("Enter your text:", placeholder="To be or not to be...")
    
    if st.button("Predict Next Word"):
        if input_text.strip():
            with st.spinner("Analyzing text..."):
                next_word = predict_next_word(model, tokenizer, input_text)
                
            if next_word:
                st.markdown(f"""
                    <div class="prediction-container">
                        <p style="margin-bottom: 5px; color: #666;">The predicted next word is:</p>
                        <p class="prediction-text">{next_word}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Could not predict the next word. Try a different phrase.")
        else:
            st.info("Please enter some text to get a prediction.")
else:
    st.error("Model or tokenizer not found. Please ensure they are in the 'models/' directory.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
