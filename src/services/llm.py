from src.config import api_key,folder_id
from src.services.templates.prompt import get_prompt
from yandex_ai_studio_sdk import AIStudio

def get_answer_yandex(review_text: str,name: str,product: str)->str:
    """Функция,которая возвращает ответ LLM на отзыв

    Args:
        review_text (str): Текст отзыва
        name (str): Имя пользователя, оставившего отзыв
        product (str): Название продукта, на который оставлен отзыв

    Returns:
        str: Текст ответа
    """
    try:
        sdk = AIStudio(
            folder_id=folder_id,
            auth=api_key
        )
        model = (
            sdk.models.completions("yandexgpt").configure(
                temperature=0.6,
                max_tokens = 300
            )
        )
        result = model.run(get_prompt(review_text,name,product))
        if result and len(result)>0:
            return result[0].text
        else:
            return 'Не удалось получить ответ от модели: пустой реузльтат'
    except Exception as e:
        print(f"Ошибка при запросе к YandexGPT: {e}")


