# text_features/cleaning.py
# Cleaning functions for basic and semantic text preprocessing.

import html
import unicodedata
import pandas as pd
import re
import pymorphy3
from razdel import tokenize
from functools import lru_cache
from typing import List
from .constants import (
    INVISIBLE_RE, CONTROL_RE, TAG_RE, WS_RE, URL_PATTERN,
    COMBINED_PATTERN, STOP_WORDS, timing
)

# Initialize pymorphy3 analyzer
morph = pymorphy3.MorphAnalyzer()

# Cache lemmatization results
@lru_cache(maxsize=10000)
def cached_lemmatize(token: str) -> str:
    """Cached lemmatization using pymorphy3."""
    return morph.parse(token)[0].normal_form

# --- Basic cleaning functions ---
def unescape_html(text: str) -> str:
    """Unescape HTML entities in text."""
    return html.unescape(text) if isinstance(text, str) else ""

def replace_specific_tags(text: str) -> str:
    """Replace specific HTML tags (br, p, li, ul, ol, div, span) with spaces."""
    return re.sub(r"(?i)</?(br|p|li|ul|ol|div|span)\b[^>]*>", " ", text) if isinstance(text, str) else text

def remove_html_tags(text: str) -> str:
    """Remove all HTML tags from text."""
    return TAG_RE.sub(" ", text) if isinstance(text, str) else text

def remove_invisible_chars(text: str) -> str:
    """Remove invisible Unicode characters."""
    return INVISIBLE_RE.sub("", text) if isinstance(text, str) else text

def replace_control_chars(text: str) -> str:
    """Replace control characters with spaces."""
    return CONTROL_RE.sub(" ", text) if isinstance(text, str) else text

def replace_special_chars(text: str) -> str:
    """Replace specific special characters (e.g., bullets) with spaces."""
    return re.sub(r"[•‣‥∙]", " ", text) if isinstance(text, str) else text

def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespaces into a single space and strip."""
    return WS_RE.sub(" ", text).strip() if isinstance(text, str) else text

def normalize_urls(text: str) -> str:
    """Normalize URLs by removing query parameters."""
    def normalize_url(match):
        url = match.group(0)
        return url.split("?")[0]
    return URL_PATTERN.sub(normalize_url, text) if isinstance(text, str) else text

# --- Dictionary of basic cleaning functions ---
CLEANING_FUNCTIONS = {
    "unescape_html": unescape_html,
    "replace_specific_tags": replace_specific_tags,
    "remove_html_tags": remove_html_tags,
    "remove_invisible_chars": remove_invisible_chars,
    "replace_control_chars": replace_control_chars,
    "replace_special_chars": replace_special_chars,
    "collapse_whitespace": collapse_whitespace,
    "normalize_urls": normalize_urls,
}

# --- Main basic cleaner ---
def basic_clean_text(text: str) -> str:
    """
    Apply a series of basic cleaning functions to the input text in sequence.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    result = text
    for name, func in CLEANING_FUNCTIONS.items():
        result = func(result)
    return result

@timing
def create_basic_cleaned_text(text_series: pd.Series) -> pd.Series:
    """
    Apply basic cleaning to a pandas Series of texts.
    
    Args:
        text_series (pd.Series): Input Series of texts.
    
    Returns:
        pd.Series: Cleaned text Series with the same index.
    """
    return text_series.apply(basic_clean_text)

# --- Semantic cleaning functions ---
def lowercase_text(series: pd.Series) -> pd.Series:
    """Convert text to lowercase (vectorized)."""
    return series.str.lower()

def normalize_unicode(series: pd.Series) -> pd.Series:
    """Normalize Unicode characters to NFKC form (vectorized)."""
    return series.apply(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else "")

def remove_combined_patterns(series: pd.Series) -> pd.Series:
    """Remove URLs, emojis, numbers, and punctuation in one pass (vectorized)."""
    return series.str.replace(COMBINED_PATTERN, " ", regex=True)

def collapse_whitespace_semantic(series: pd.Series) -> pd.Series:
    """Collapse multiple whitespaces into a single space (vectorized)."""
    return series.str.replace(WS_RE, " ", regex=True).str.strip()

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words using razdel."""
    if not isinstance(text, str) or text == "":
        return []
    return [token.text for token in tokenize(text)]

def lemmatize_and_filter(tokens: List[str]) -> List[str]:
    """Lemmatize tokens, filter out non-alphabetic tokens and stop words."""
    if not tokens:
        return []
    return [
        cached_lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in STOP_WORDS
    ]

def remove_consecutive_duplicates(tokens: List[str]) -> List[str]:
    """Remove consecutive duplicate words from the token list."""
    if not tokens:
        return []
    result = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i-1]:
            result.append(tokens[i])
    return result

def join_tokens(tokens: List[str]) -> str:
    """Join tokens back into a string with spaces."""
    return " ".join(tokens) if tokens else ""

# --- Per-text semantic cleaning function ---
def clean_text_semantic_single(text: str) -> str:
    """
    Process a single text through tokenization, lemmatization, and duplicate removal.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Semantically cleaned text.
    """
    if not isinstance(text, str) or text == "":
        return ""
    
    # Tokenize
    tokens = tokenize_text(text)
    
    # Lemmatize and filter
    tokens = lemmatize_and_filter(tokens)
    
    # Remove consecutive duplicates
    tokens = remove_consecutive_duplicates(tokens)
    
    # Join tokens
    return join_tokens(tokens)

# --- Dictionary of vectorized semantic cleaning functions ---
VECTORIZED_CLEANING_FUNCTIONS = {
    "lowercase_text": lowercase_text,
    "normalize_unicode": normalize_unicode,
    "remove_combined_patterns": remove_combined_patterns,
    "collapse_whitespace": collapse_whitespace_semantic,
}


@timing
# --- Main semantic cleaner ---
def create_semantic_cleaned_text(text_series: pd.Series) -> pd.Series:
    """
    Apply semantic cleaning (lowercase, normalize, remove patterns, lemmatize, etc.) to a pandas Series.
    
    Args:
        text_series (pd.Series): Input Series of texts.
    
    Returns:
        pd.Series: Semantically cleaned text Series with the same index.
    """
    if not isinstance(text_series, pd.Series):
        raise ValueError("Input must be a pandas Series")

    # Apply vectorized operations
    result = text_series
    for name, func in VECTORIZED_CLEANING_FUNCTIONS.items():
        result = func(result)
    
    # Apply tokenization, lemmatization, and duplicate removal
    cleaned_texts = result.apply(clean_text_semantic_single)
    
    # Return as Series with original index
    return pd.Series(cleaned_texts, index=text_series.index)
