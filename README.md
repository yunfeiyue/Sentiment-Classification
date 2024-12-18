# Sentiment-Classification
sentiment classification based on multiple deep learning models

This repository provides implementations of various deep learning models for sentiment classification tasks. The models include:

- **TextCNN**
- **LSTM**
- **BiLSTM**
- **LSTM with Attention**
- **BiLSTM with Attention**
- **BERT**
- **RoBERTa**

Each model is implemented using PyTorch and designed to classify sentiment polarity (positive/negative) from text data.

---

## Supported Datasets

The following datasets can be used with the provided code:

### 1. **IMDB Dataset**
- **Description**: The IMDB dataset contains 50,000 movie reviews, labeled as positive or negative.
- **Source**: The dataset can be loaded directly using `torchtext` or the `datasets` library from Hugging Face.
- **Where to Access**:
  - For `torchtext`: Automatically downloaded when using the `datasets.IMDB.splits` method.
  - For `datasets`: Automatically downloaded using the `datasets.load_dataset("imdb")` method.

### 2. **Amazon Reviews Dataset**
- **Description**: A large-scale dataset containing customer reviews and ratings from Amazon.
- **Source**: Available from [AWS Open Data Registry](https://registry.opendata.aws/amazon-reviews/).
- **How to Download**:
  - Use the `datasets` library: `datasets.load_dataset("amazon_polarity")`.

### 3. **Yelp Reviews Dataset**
- **Description**: Yelp dataset containing customer reviews with sentiment labels (positive/negative).
- **Source**: Available from the [Yelp Dataset Challenge](https://www.yelp.com/dataset).
- **How to Download**:
  - Use the `datasets` library: `datasets.load_dataset("yelp_polarity")`.

### 4. **Twitter Sentiment Analysis Dataset**
- **Description**: Dataset of tweets annotated with sentiment labels.
- **Source**: Various repositories, such as Kaggle ([Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)).
- **How to Download**:
  - Manually download from Kaggle or preprocess a Twitter dataset using custom scripts.

---

## Model Descriptions and Dataset Usage

### **TextCNN**
- **Dataset**: IMDB (via `torchtext`)
- **Code Location**: `textcnn_sentiment.py`
- **Usage**:
  - Automatically loads and preprocesses IMDB data using `torchtext`.
  - Embeddings are initialized with GloVe vectors.

### **LSTM and BiLSTM**
- **Dataset**: IMDB (via `torchtext`)
- **Code Location**: `lstm_sentiment.py`, `bilstm_sentiment.py`
- **Usage**:
  - Processes data using `torchtext` and tokenizes using SpaCy.
  - Can be trained on other datasets by replacing the dataset loading part with any binary sentiment dataset compatible with `torchtext`.

### **LSTM with Attention and BiLSTM with Attention**
- **Dataset**: IMDB (via `torchtext`)
- **Code Location**: `lstm_attention_sentiment.py`, `bilstm_attention_sentiment.py`
- **Usage**:
  - Extends LSTM/BiLSTM models with an attention mechanism.

### **BERT**
- **Dataset**: IMDB (via `datasets`)
- **Code Location**: `bert_sentiment.py`
- **Usage**:
  - Loads IMDB dataset using the Hugging Face `datasets` library.
  - Tokenizes using the BERT tokenizer (`BertTokenizer` from `transformers`).
  - Adapts the pre-trained BERT model (`BertForSequenceClassification`) for sentiment classification.

### **RoBERTa**
- **Dataset**: IMDB (via `datasets`)
- **Code Location**: `roberta_sentiment.py`
- **Usage**:
  - Similar to the BERT implementation but uses the RoBERTa tokenizer (`RobertaTokenizer`) and model (`RobertaForSequenceClassification`).

---

## Prerequisites

### Install Required Libraries
- Python 3.8+
- PyTorch 1.10+
- Hugging Face Transformers Library
- `torchtext` for TextCNN, LSTM, BiLSTM
- `datasets` library for BERT and RoBERTa
- SpaCy (for tokenization)

### Installation Commands
```bash
pip install torch torchvision torchtext transformers datasets spacy
python -m spacy download en_core_web_sm
```

---

## How to Run

1. **Select a Model**:
   - Each model is stored in a separate file (e.g., `textcnn_sentiment.py`, `bert_sentiment.py`).
   - Open the file corresponding to the model you want to train/test.

2. **Choose a Dataset**:
   - By default, all scripts use the IMDB dataset.
   - To use a different dataset, modify the dataset loading and preprocessing sections.

3. **Run the Script**:
   ```bash
   python <script_name>.py
   ```

4. **Evaluate the Model**:
   - After training, the script will evaluate the model on the test dataset and print the accuracy and loss.

---

## Notes on Datasets

- The IMDB dataset is binary-labeled (positive/negative). Ensure other datasets you use are similarly formatted.
- For other datasets, ensure compatibility with the tokenizer and input pipeline of the model.
- Hugging Face `datasets` library supports many datasets out of the box, and you can replace the dataset name (e.g., `"imdb"`) to experiment with others.
