import random
from src.services.classifier import predict_sentiment
from src.services.templates.answers import answers
from src.services.llm import get_answer_yandex


def generate_response(review_text,name,product):
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
