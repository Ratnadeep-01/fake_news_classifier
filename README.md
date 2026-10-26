# Fake News Detection using Bidirectional LSTM

A Natural Language Processing (NLP) deep learning project that classifies news articles as **Real** or **Fake** using TensorFlow and Keras.

## Project Overview
The pipeline processes news headlines and body text to detect misinformation. It implements text preprocessing, tokenization, sequence padding, and a **Bidirectional Long Short-Term Memory (BiLSTM)** network to capture contextual information from text sequences in both directions.

## Repository Structure
- **[`fakenews.ipynb`](fakenews.ipynb)**: End-to-end Jupyter Notebook covering data cleaning, text preprocessing (stopword removal), tokenization, BiLSTM model building, training with early stopping, and performance evaluation.
- **[`fake_news_dataset.csv`](fake_news_dataset.csv)**: Dataset containing news records with fields: `title`, `text`, `author`, `source`, `category`, and `label` (`real` / `fake`).

## Model Pipeline
1. **Text Preprocessing**: Merges `title` and `text`, normalizes characters, and removes stopwords.
2. **Tokenization & Padding**: Converts text to integer token sequences and pads them to a uniform length.
3. **Architecture**:
   - **Embedding Layer**: Maps token IDs to dense vectors.
   - **SpatialDropout1D**: Regularizes embedding feature representations.
   - **Bidirectional LSTM**: Captures bidirectional sequential patterns.
   - **Dense & Dropout Layers**: High-level feature representation and regularization.
   - **Sigmoid Output**: Binary classification (`0: Fake`, `1: Real`).

## Requirements
- Python 3.8+
- TensorFlow / Keras
- Pandas
- NumPy
- NLTK
- Scikit-learn
