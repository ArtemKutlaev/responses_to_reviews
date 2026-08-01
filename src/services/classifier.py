import joblib
from src.config_path import VECTORIZER_PATH,MODEL_PATH
from src.ml.utils import preprocess_text


try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    model = None
    vectorizer = None
    print(f"Ошибка при загрузке модели или векторизатора: {e}")

def predict_sentiment(text: str) -> str:
    """Функция, которая определяет тональность текста 
    Args:
        text (str): Текст отзыва

    Returns:
        _type_: Тональность отзыва ('Negative', 'Neutral','Positive')
    """
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict_proba(vectorized_text)[0]
    pos_confidence = prediction[2]
    neg_confidence = prediction[0]
    
    if pos_confidence >=0.85:
        return "Positive"
    elif neg_confidence >= 0.7:
        return "Negative"
    else:
        return "Neutral"    
