# 💬 Tweet Sentiment Analyzer using LSTM and GRU

---

## Objective

To develop an AI-powered tool that classifies tweets into **Positive**, **Negative**, and **Neutral** sentiments using deep learning models (LSTM & GRU). The system enhances social media monitoring, brand analysis, and public sentiment understanding.

---

## Dataset

- **Source**: Collected from Twitter using Tweepy API and pre-cleaned CSV
- **Classes**: Positive, Negative, Neutral
- **Text Format**: Preprocessed to remove URLs, emojis, stopwords, and punctuations
- **Tokenization**: Using Keras Tokenizer + Padding

---

## Model Architectures

### 1. LSTM (Long Short-Term Memory)
- Embedding layer (Word2Vec / Glove optional)
- Single-layer LSTM (64 units)
- Dense output with softmax
- Accuracy: ~**84%** on test set

### 2. GRU (Gated Recurrent Unit)
- Similar architecture with GRU replacing LSTM
- Lighter and faster convergence
- Accuracy: ~**82%**

Both models include:
- Dropout for regularization
- EarlyStopping to prevent overfitting
- Class weights for imbalance handling

---

## Performance Metrics

| Model | Accuracy | F1 (Positive) | F1 (Negative) | F1 (Neutral) |
|-------|----------|---------------|---------------|--------------|
| LSTM  | 84%      | 0.86          | 0.81          | 0.79         |
| GRU   | 82%      | 0.84          | 0.80          | 0.76         |

---

## Preprocessing Steps

- Lowercasing
- Removing punctuation, hashtags, mentions, and links
- Tokenization and padding to uniform sequence length
- Label encoding for sentiment classes

---

## Tools & Technologies

| Category        | Tools Used                      |
|-----------------|----------------------------------|
| Programming     | Python                          |
| NLP             | NLTK, TextBlob, Regex           |
| DL Frameworks   | TensorFlow, Keras               |
| Visualization   | Matplotlib, Seaborn             |
| Deployment      | Flask (optional future scope)   |

---

## UI Features (Future Scope)

- Textbox for live tweet input
- Emoji-based output 🌞 😐 😡
- Upload CSV for batch sentiment analysis
- REST API endpoint for model interaction
