import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(file_path: str = "data/generated_1000_support_tickets.csv") -> pd.DataFrame:
    """
    Load the email subjects dataset from a CSV file.
    Expects a CSV with columns: 'text' and 'label'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    #determine file extension and load accordingly
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please use CSV, XLSX, or XLS."
        )

    # Validate required columns
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"File must contain the columns: {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    return df


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the DataFrame into train and test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep label distribution
    )
    return X_train, X_test, y_train, y_test


def create_vectorizer() -> TfidfVectorizer:
    """
    Create and configure a TF-IDF vectorizer.
    This converts text into numeric features for the model.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        # You can tweak options later, e.g.:
        # ngram_range=(1, 2)
    )
    return vectorizer


def save_model(model, vectorizer, path: str = "models/text_classifier.pkl") -> None:
    """
    Save the trained model and vectorizer to disk as a single .pkl file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)


def load_model(path: str = "models/text_classifier.pkl"):
    """
    Load the trained model and vectorizer from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]
