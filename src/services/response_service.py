import random
from src.services.classifier import predict_sentiment
from src.services.templates.answers import answers
from src.services.llm import get_answer_yandex


def generate_response(review_text: str,name: str,product: str)->dict:
    """Функция, которая определяет,куда пойдет запрос в зависимости от полученной тональности

    Args:
        review_text (str): Текст отзыва
        name (str): Имя пользователя, оставившего отзыв
        product (str): Название продукта, на который оставлен отзыв

    Returns:
        dict: Словарь c результатом обработки
            -sentiment(str): Тональность отзыва (Positive,Neutral,Negative)
            -reply(str): Текст сгенерированного или шаблонного ответа
            -source(str): Источник ответа (template,llm)
    """
    sentiment = predict_sentiment(review_text)
    
    if sentiment == 'Positive':
        reply = random.choice(answers)
        source = 'template'
    else: 
        reply = get_answer_yandex(review_text,name,product)
        source = 'llm'
    
    return {
        'sentiment': sentiment,
        'reply':reply,
        'source':source
    }
