import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def remove_duplicates(df):
    """
    Removes duplicate rows from a dataframe.

    Args:
        df (DataFrame): Input dataframe.

    Returns:
        DataFrame: Dataframe with duplicate rows removed.
    """
    df = df.drop_duplicates()
    return df


def clean_text(text):
    """
    Cleans raw text by applying standard preprocessing steps.

    Steps performed:
        1. Lowercasing
        2. HTML tag removal
        3. Negation handling (e.g., 'not good' → 'not_good')
        4. Punctuation removal
        5. Digit removal
        6. Extra whitespace removal

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\bnot\s+(\w+)', r'not_\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def download_nltk_resources():
    """
    Downloads required NLTK resources for text preprocessing.
    Should be called once before preprocessing begins.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


stop_words = set(stopwords.words('english'))


def text_preprocessing_mlp(text):
    """
    Preprocesses text for MLP-based models using lemmatization
    and stopword removal.

    Args:
        text (str): Cleaned input text.

    Returns:
        str: Preprocessed text suitable for TF-IDF vectorization.
    """
    lemmatizer = WordNetLemmatizer()
    tokens_list = text.split()
    filtered_tokens = [word for word in tokens_list if ((word not in stop_words) and (len(word) > 2))]
    tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(tokens)


def tfidf_vectorization(df_col, max_features):
    """
    Converts text into TF-IDF feature vectors.

    Args:
        df_col (Series or list): Preprocessed text data.
        max_features (int): Maximum number of TF-IDF features.

    Returns:
        tuple:
            - sparse matrix: TF-IDF feature matrix
            - TfidfVectorizer: fitted vectorizer
    """
    tfidf = TfidfVectorizer(max_features=max_features)
    result = tfidf.fit_transform(df_col)
    return result, tfidf


def split_dataset(X, df, test_split_ratio=0.2, rand_state=42):
    """
    Splits data into training and testing sets.

    Args:
        X: Feature matrix.
        df: DataFrame containing 'label' column.
        test_split_ratio (float): Proportion of test data.
        rand_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=rand_state
    )
    return X_train, X_test, y_train, y_test


def tokenize_text(df_col, num_words=600, max_len=200):
    """
    Tokenizes and pads text sequences for sequence-based models
    such as CNNs.

    Args:
        df_col (Series or list): Cleaned text data.
        num_words (int): Maximum vocabulary size.
        max_len (int): Maximum sequence length.

    Returns:
        tuple:
            - ndarray: Padded sequences
            - Tokenizer: fitted tokenizer
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df_col)
    sequences = tokenizer.texts_to_sequences(df_col)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding='post', truncating='post'
    )
    return padded_sequences, tokenizer
