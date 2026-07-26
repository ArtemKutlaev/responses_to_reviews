import streamlit as st
from src.services.response_service import generate_response

st.title('Система автоматических ответов на отзывы')

name = st.text_input('Имя покупателя')
product = st.text_input('Название товара')
review_text = st.text_area('Введите текст отзыва')

if st.button('Обработать отзыв'):
    if review_text == '':
        st.warning('Пожалуйста, введите текст отзыва!')
    else:
        with st.spinner('Анализируем и пишем ответ...'):
            response = generate_response(review_text, name,product)
        st.write(response['reply'])
    
