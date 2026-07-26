import sqlite3
import pandas as pd

def init_db()->None:
    """Инициализирует базу данных: создает таблицу reviews_db,
    если она еще не существует
    """
    db = sqlite3.connect('data/database.db')
    c = db.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS reviews_db(
                  reviewer TEXT,
                  rating INTEGER,
                  comment TEXT
              )
    ''' )
    db.commit()
    db.close()

if __name__ == '__main__':
    init_db()
    
def is_valid_comment(text)-> bool:
    """Проверяет, является ли комментарий валидным 
    (содержит больше двух уникальных слов).
    """
    if not isinstance(text, str):
        return False
    if len(set(text.lower().split())) <= 2:
        return False
    return True

with sqlite3.connect('data/database.db') as db:
    data = pd.read_excel('Ваша ссылка на excel')
    data = data.dropna(subset=["comment"])
    data = data.drop_duplicates(subset=["comment"])
    data = data[data['comment'].apply(is_valid_comment)]
    words = ['Достоинства:', 'Недостатки:', 'Комментарий:', 'Первоначальный отзыв']
    for h in words:
        data["comment"] = data["comment"].str.replace(h, f' {h} ', regex=False)
   
    target_size = 2000
    data_balanced = []
    for rating_value,group in data.groupby('rating'):
        if len(group)>target_size:
            group = group.sample(n=target_size,random_state=1)
        
        data_balanced.append(group)
    data_balanced = pd.concat(data_balanced,ignore_index=True)
    
    data_balanced = data_balanced.sample(frac=1, random_state=1).reset_index(drop=True)
    data_balanced.to_sql('reviews_db', con=db, if_exists='replace', index=False)
