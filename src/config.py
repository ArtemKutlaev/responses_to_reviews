from dotenv import load_dotenv
import os

#Загружаем переменные окружения из файла .env
load_dotenv()

#Получаем учетные данные для Yandex GPT
api_key = os.getenv('YANDEX_GPT_API')
folder_id = os.getenv('FOLDER_ID')