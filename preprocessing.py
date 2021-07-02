import pandas as pd
import nltk
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces, split_alphanum, strip_short, strip_numeric
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv('datasets/sms_spam.csv', header=0, encoding='latin-1')
    labels = df.iloc[:, 0]
    texts = df.iloc[:, 1]
    return texts.to_numpy(), labels.to_numpy()

def preprocessing_text(texts):
    # To lowercase
    texts = [text.lower() for text in texts]

    # Remove punctiation
    texts = [strip_non_alphanum(text) for text in texts]

    # Separate number and word
    texts = [split_alphanum(text) for text in texts]

    # Remove one letter word
    texts = [strip_short(text) for text in texts]

    # Remove number
    texts = [strip_numeric(text) for text in texts]

    # Remove multiple whitespace
    texts = [strip_multiple_whitespaces(text) for text in texts]

    # Remove stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    texts = [' '.join([word for word in text.split() if word not in stop_words]) for text in texts]

    return texts

def preprocessing_label(labels):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels
