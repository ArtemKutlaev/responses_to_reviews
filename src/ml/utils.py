import re
import pymorphy3
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian')) - {'не', 'ни'}

morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    #токенизация
    text = text.lower()
    text = re.sub(r'[^а-яё\s-]', '', text)
    text = text.split()
    #лемматизация и удаление стоп слов
    text = [morph.parse(word)[0].normal_form for word in text if word not in stop_words]
    # Объединяем обратно в строку для векторизатора
    return ' '.join(text)