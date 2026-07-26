import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ml.utils import preprocess_text
from src.config_path import DB_PATH


def get_prepared_data():
    db = sqlite3.connect(DB_PATH) 
    data = pd.read_sql_query(sql="SELECT * FROM reviews_db", con=db)
    db.close()

    data['clean_text_str'] = data['Comment'].apply(preprocess_text)

    # векторизация
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000,max_df=0.9)
    X = vectorizer.fit_transform(data['clean_text_str'])

    def rating_to_sentiment(rating):
        #отрицательный
        if rating <= 2: return 0
        #нейтральный
        elif rating == 3: return 1
        #положительный
        else: return 2

    y = data['Rating'].apply(rating_to_sentiment)
    return X,y, vectorizer, data['Comment']
    
    
