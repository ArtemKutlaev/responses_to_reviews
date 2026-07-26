import re
import pymorphy3
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian')) - {'не', 'ни'}

morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    """Функция, которая делает токенизацию, лемматизацию
    Args:
        text (str): Текст отзыва

    Returns:
        str: Обработаный текст отзыва
    """
    text = text.lower()
    text = re.sub(r'[^а-яё\s-]', '', text)
    text = text.split()
    text = [morph.parse(word)[0].normal_form for word in text if word not in stop_words]
    return ' '.join(text)