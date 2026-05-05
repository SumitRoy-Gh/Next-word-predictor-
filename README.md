# Next Word Predictor

A simple deep learning application that predicts the next word in a sequence using an LSTM (Long Short-Term Memory) neural network. The model is trained on Shakespeare's "Hamlet".

## Features
- **Deep Learning Model**: Built with TensorFlow/Keras using LSTM layers.
- **Interactive UI**: A sleek Streamlit interface for real-time predictions.
- **Preprocessing**: Tokenization and sequence padding using the trained tokenizer.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SumitRoy-Gh/Next-word-predictor-.git
   cd Next-word-predictor-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure
- `models/`: Contains the trained `.h5` model and `.pickle` tokenizer.
- `LSTM RNN/`: Contains the experimentation notebook and source text.
- `app.py`: The Streamlit web application.