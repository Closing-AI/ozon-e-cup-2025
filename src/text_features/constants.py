# text_features/constants.py
# Shared constants for regex patterns and word lists.

import re
import nltk
from pathlib import Path
from functools import wraps
from nltk.corpus import stopwords
import time

# --- Word lists ---
MESSENGERS = [
    "whatsapp", "telegram", "viber", "wechat", "signal", "icq", "вк",
    "вконтакте", "телеграм", " телега", "тг", "тгк", "instagram",
    "инст", "инста", "инстаграм"
]
SUSPICIOUS_WORDS = {
    "реплика", "копия", "аналог", "подделка", "реселл", "фейк",
    "контрафакт", "дешево", "скидка", "акция", "оригинал", "коробка", "оригинальный",
    "позвонить", "написать", "перепродажа"
}
BRANDS = {
    "nike", "adidas", "gucci", "apple", "samsung", "rolex",
    "louisvuitton", "chanel", "prada", "reebok", "philips", "apple", "logitech"
}
URGENCY_WORDS = {
    "срочно", "быстро", "спешить", "поспешить", "ограниченный",
    "последний", "сегодня", "немедленно"
}

# --- Regex patterns ---
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+7|8)\s*[\-]?\s*\(?\d{3}\)?\s*[\-]?\s*\d{3}\s*[\-]?\s*\d{2}\s*[\-]?\s*\d{2}(?!\d)"
)
SKU_PATTERN = re.compile(r"\b(?:[A-ZА-Я0-9]{5,15})\b")
PRICE_PATTERN = re.compile(r"\b\d{1,6}\s*(?:руб|р|₽)\b", re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF'
    r'\U00002702-\U000027B0'
    r'\U000024C2-\U0001F251]+'
)
INVISIBLE_RE = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF\uFFF9-\uFFFB]"
)
CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
COMBINED_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|"
    r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+|'
    r"\b\d+\b|[^\w\s])",
    re.IGNORECASE
)

# --- Stopwords loading ---
def load_stopwords() -> set:
    """
    Load Russian stopwords from a local file if it exists, otherwise download from NLTK and save.
    
    Returns:
        set: Set of Russian stopwords.
    """
    stopwords_file = Path("data/stopwords_russian.txt")
    
    # Check if stopwords file exists
    if stopwords_file.exists():
        with open(stopwords_file, "r", encoding="utf-8") as f:
            return set(word.strip() for word in f.readlines())
    
    # If file doesn't exist, try to download from NLTK
    try:
        nltk.download('stopwords', quiet=True)
        stopwords_set = set(stopwords.words("russian"))
        
        # Create data directory if it doesn't exist
        stopwords_file.parent.mkdir(exist_ok=True)
        
        # Save stopwords to file
        with open(stopwords_file, "w", encoding="utf-8") as f:
            for word in stopwords_set:
                f.write(word + "\n")
        
        return stopwords_set
    except Exception as e:
        print(f"Warning: Failed to download NLTK stopwords: {e}. Returning empty set.")
        return set()

STOP_WORDS = load_stopwords()


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper
