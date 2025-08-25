# text_features/features.py
# Feature extraction functions for text data processing.

import re
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Union, Tuple, Optional, List
from .constants import (
    URL_PATTERN, PHONE_PATTERN, SKU_PATTERN, PRICE_PATTERN, EMOJI_PATTERN,
    MESSENGERS, SUSPICIOUS_WORDS, BRANDS, URGENCY_WORDS, timing
)
from .cleaning import create_basic_cleaned_text, create_semantic_cleaned_text



# --- Basic feature functions ---
def count_capslock_words(text: str) -> int:
    """Count words in all caps (at least 6 characters, alphabetic only)."""
    return sum(
        1
        for word in text.split()
        if word.isalpha()
        and word.isupper()
        and len(word) > 5
        and not any(ch.isdigit() for ch in word)
    )

def has_url(text: str) -> int:
    """Check if text contains a URL."""
    return int(bool(URL_PATTERN.search(text)))

def has_phone(text: str) -> int:
    """Check if text contains a phone number."""
    return int(bool(PHONE_PATTERN.search(text)))

def has_messenger(text: str) -> int:
    """Check if text contains messenger-related words."""
    return int(any(re.search(rf"\b{re.escape(m)}\b", text.lower()) for m in MESSENGERS))

def has_sku(text: str) -> int:
    """Check if text contains a SKU (5-15 alphanumeric characters)."""
    return int(bool(SKU_PATTERN.search(text)))

def desc_len_chars(text: str) -> int:
    """Count characters in text."""
    return len(text)

def desc_len_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def capslock_word_count(text: str) -> int:
    """Count capslock words (wrapper for count_capslock_words)."""
    return count_capslock_words(text)

def exclamation_count(text: str) -> int:
    """Count exclamation marks (including double exclamation)."""
    return text.count('!') + text.count('‼')

def question_count(text: str) -> int:
    """Count question marks."""
    return text.count('?')

def avg_word_length(text: str) -> float:
    """Calculate average word length."""
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0.0

def has_price(text: str) -> int:
    """Check if text contains a price (rubles)."""
    return int(bool(PRICE_PATTERN.search(text)))

def upper_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters to total letters."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0

def has_emoji(text: str) -> int:
    """Check if text contains emojis."""
    return int(bool(EMOJI_PATTERN.search(text)))

def emoji_count(text: str) -> int:
    """Count emojis in text."""
    return len(EMOJI_PATTERN.findall(text))

# --- Semantic feature functions ---
def has_suspicious_words(text: str) -> int:
    """Check if text contains suspicious words."""
    words = text.split()
    return int(any(word in SUSPICIOUS_WORDS for word in words))

def suspicious_word_count(text: str) -> int:
    """Count occurrences of suspicious words."""
    words = text.split()
    return sum(1 for word in words if word in SUSPICIOUS_WORDS)

def has_brand(text: str) -> int:
    """Check if text contains brand names."""
    words = text.split()
    return int(any(word in BRANDS for word in words))

def brand_count(text: str) -> int:
    """Count occurrences of brand names."""
    words = text.split()
    return sum(1 for word in words if word in BRANDS)

def has_urgency_words(text: str) -> int:
    """Check if text contains urgency-related words."""
    words = text.split()
    return int(any(word in URGENCY_WORDS for word in words))

def urgency_word_count(text: str) -> int:
    """Count occurrences of urgency-related words."""
    words = text.split()
    return sum(1 for word in words if word in URGENCY_WORDS)

def unique_word_ratio(text: str) -> float:
    """Calculate ratio of unique words to total words."""
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0

# --- Dictionaries of feature functions ---
FEATURE_FUNCTIONS = {
    "has_url": has_url,
    "has_phone": has_phone,  # Included despite being commented out in original
    "has_messenger": has_messenger,
    "has_sku": has_sku,
    "desc_len_chars": desc_len_chars,
    "desc_len_words": desc_len_words,
    "capslock_word_count": capslock_word_count,
    "exclamation_count": exclamation_count,
    "question_count": question_count,
    "avg_word_length": avg_word_length,
    "has_price": has_price,
    "upper_ratio": upper_ratio,
    "has_emoji": has_emoji,
    "emoji_count": emoji_count,
}

SEMANTIC_FEATURE_FUNCTIONS = {
    "has_suspicious_words": has_suspicious_words,
    "suspicious_word_count": suspicious_word_count,
    "has_brand": has_brand,  # Included despite being commented out in original
    "brand_count": brand_count,
    "has_urgency_words": has_urgency_words,
    "urgency_word_count": urgency_word_count,
    "unique_word_ratio": unique_word_ratio,
}

# --- Basic feature extractor ---
def extract_basic_text_features(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract basic text features from a single text string.
    
    Args:
        text (str): Input text to process.
        
    Returns:
        Dict[str, Union[int, float]]: Dictionary of feature names and their values.
    """
    if not isinstance(text, str):
        return {name: 0 if "ratio" not in name and "avg" not in name else 0.0 for name in FEATURE_FUNCTIONS}
    
    return {name: func(text) for name, func in FEATURE_FUNCTIONS.items()}

# --- Semantic feature extractor ---
def extract_semantic_features(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract semantic text features from a single text string.
    
    Args:
        text (str): Input text to process.
        
    Returns:
        Dict[str, Union[int, float]]: Dictionary of semantic feature names and their values.
    """
    if not isinstance(text, str) or not text.strip():
        return {name: 0 if "ratio" not in name else 0.0 for name in SEMANTIC_FEATURE_FUNCTIONS}
    
    return {name: func(text) for name, func in SEMANTIC_FEATURE_FUNCTIONS.items()}

@timing
# --- Feature creation functions ---
def create_basic_text_features(text_series: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame of basic text features from a Series of texts.
    
    Args:
        text_series (pd.Series): Input Series of texts.
        
    Returns:
        pd.DataFrame: DataFrame with extracted features, preserving the original index.
    """
    features_df = text_series.apply(extract_basic_text_features).apply(pd.Series)
    features_df.index = text_series.index
    return features_df

@timing
def create_semantic_features(text_series: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame of semantic text features from a Series of pre-cleaned texts.
    
    Args:
        text_series (pd.Series): Input Series of texts.
        
    Returns:
        pd.DataFrame: DataFrame with semantic features, preserving the original index.
    """
    features_df = text_series.apply(extract_semantic_features).apply(pd.Series)
    features_df.index = text_series.index
    return features_df

@timing
def create_tfidf_features(
    texts_char: pd.Series,
    texts_word: pd.Series,
    mode: str = 'train',
    tfidf_char_vectorizer: Optional[TfidfVectorizer] = None,
    tfidf_word_vectorizer: Optional[TfidfVectorizer] = None,
    max_tfidf_features: int = 1000,
    char_ngram_range: Tuple[int, int] = (3, 5),
    word_ngram_range: Tuple[int, int] = (1, 3)
) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer, TfidfVectorizer]:
    """
    Create TF-IDF features for character and word n-grams from two separate pandas Series of texts.
    
    Args:
        texts_char (pd.Series): Series of cleaned text for character-level TF-IDF.
        texts_word (pd.Series): Series of cleaned text for word-level TF-IDF.
        mode (str): 'train' to fit and transform, 'test' to only transform.
        tfidf_char_vectorizer: Fitted TfidfVectorizer for character n-grams (required for test mode).
        tfidf_word_vectorizer: Fitted TfidfVectorizer for word n-grams (required for test mode).
        max_tfidf_features (int): Maximum number of TF-IDF features to generate.
        char_ngram_range (tuple): N-gram range for character-level TF-IDF.
        word_ngram_range (tuple): N-gram range for word-level TF-IDF.
    
    Returns:
        Tuple: (tfidf_char_df, tfidf_word_df, tfidf_char_vectorizer, tfidf_word_vectorizer)
            - tfidf_char_df: DataFrame with character TF-IDF features.
            - tfidf_word_df: DataFrame with word TF-IDF features.
            - tfidf_char_vectorizer: Fitted or input character vectorizer.
            - tfidf_word_vectorizer: Fitted or input word vectorizer.
    """
    if mode not in ['train', 'test']:
        raise ValueError("Mode must be 'train' or 'test'")
    if mode == 'test' and (tfidf_char_vectorizer is None or tfidf_word_vectorizer is None):
        raise ValueError("Fitted vectorizers must be provided for test mode")
    if not texts_char.index.equals(texts_word.index):
        raise ValueError("Indices of texts_char and texts_word must match")

    if mode == 'train':
        tfidf_char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            max_features=max_tfidf_features
        )
        tfidf_word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=max_tfidf_features
        )

    tfidf_char_matrix = tfidf_char_vectorizer.fit_transform(texts_char) if mode == 'train' else tfidf_char_vectorizer.transform(texts_char)
    tfidf_word_matrix = tfidf_word_vectorizer.fit_transform(texts_word) if mode == 'train' else tfidf_word_vectorizer.transform(texts_word)

    tfidf_char_df = pd.DataFrame(
        tfidf_char_matrix.toarray(),
        columns=[f"tfidf_char_{i}" for i in range(tfidf_char_matrix.shape[1])],
        index=texts_char.index
    )
    tfidf_word_df = pd.DataFrame(
        tfidf_word_matrix.toarray(),
        columns=[f"tfidf_word_{i}" for i in range(tfidf_word_matrix.shape[1])],
        index=texts_word.index
    )

    return tfidf_char_df, tfidf_word_df, tfidf_char_vectorizer, tfidf_word_vectorizer

@timing
def create_flags_features(df: pd.DataFrame, flag_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create binary flag features indicating None/NaN values for specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        flag_columns (list): List of column names to check for None/NaN.
                           Defaults to ['brand_name', 'description', 'name_rus', 'CommercialTypeName4'].
    
    Returns:
        pd.DataFrame: DataFrame with binary columns (e.g., 'is_none_<column>') where 1 indicates
                      None/NaN or missing column, 0 otherwise.
    """
    if flag_columns is None:
        flag_columns = ['brand_name', 'description', 'name_rus', 'CommercialTypeName4']
    
    none_flags = pd.DataFrame(index=df.index)
    for col in flag_columns:
        if col in df.columns:
            none_flags[f'is_none_{col}'] = df[col].isna().astype(int)
        else:
            none_flags[f'is_none_{col}'] = 1
    
    return none_flags

@timing
def create_all_text_features(
    df: pd.DataFrame,
    mode: str = 'train',
    max_tfidf_features: int = 1000,
    char_ngram_range: Tuple[int, int] = (3, 5),
    word_ngram_range: Tuple[int, int] = (1, 3),
    tfidf_char_vectorizer: Optional[TfidfVectorizer] = None,
    tfidf_word_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[pd.DataFrame, Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    """
    Create all text features: basic, semantic, TF-IDF, and flags from df['description'].
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'description' column.
        mode (str): 'train' to fit and transform, 'test' to only transform.
        max_tfidf_features (int): Maximum number of TF-IDF features.
        char_ngram_range (tuple): N-gram range for character-level TF-IDF.
        word_ngram_range (tuple): N-gram range for word-level TF-IDF.
        tfidf_char_vectorizer: Fitted TfidfVectorizer for character n-grams (test mode).
        tfidf_word_vectorizer: Fitted TfidfVectorizer for word n-grams (test mode).
    
    Returns:
        Tuple: (all_features_df, tfidf_char_vectorizer, tfidf_word_vectorizer)
            - all_features_df: DataFrame with concatenated basic, semantic, TF-IDF, and flag features.
            - tfidf_char_vectorizer: Fitted or input character vectorizer.
            - tfidf_word_vectorizer: Fitted or input word vectorizer.
    """
    if 'description' not in df.columns:
        raise ValueError("df must contain a 'description' column")
    if mode not in ['train', 'test']:
        raise ValueError("Mode must be 'train' or 'test'")
    if mode == 'test' and (tfidf_char_vectorizer is None or tfidf_word_vectorizer is None):
        raise ValueError("Fitted vectorizers must be provided for test mode")

    # Create None flags
    none_flags = create_flags_features(df)

    # Extract description
    description = df['description']

    # Basic cleaning
    basic_cleaned_description = create_basic_cleaned_text(description)

    # Basic text features
    basic_text_features = create_basic_text_features(basic_cleaned_description)

    # Semantic cleaning
    semantic_cleaned_text = create_semantic_cleaned_text(basic_cleaned_description)

    # Semantic features
    semantic_text_features = create_semantic_features(semantic_cleaned_text)

    # TF-IDF features
    tfidf_char_df, tfidf_word_df, tfidf_char_vectorizer, tfidf_word_vectorizer = create_tfidf_features(
        texts_char=basic_cleaned_description,
        texts_word=semantic_cleaned_text,
        mode=mode,
        tfidf_char_vectorizer=tfidf_char_vectorizer,
        tfidf_word_vectorizer=tfidf_word_vectorizer,
        max_tfidf_features=max_tfidf_features,
        char_ngram_range=char_ngram_range,
        word_ngram_range=word_ngram_range
    )

    # Concatenate all features
    all_features_df = pd.concat(
        [none_flags, basic_text_features, semantic_text_features, tfidf_char_df, tfidf_word_df],
        axis=1
    )

    return all_features_df, tfidf_char_vectorizer, tfidf_word_vectorizer
