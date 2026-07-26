import sqlite3
import pandas as pd

def init_db():
    db = sqlite3.connect('data/database.db')
    c = db.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS reviews_db(
                  Reviewer TEXT,
                  Rating INTEGER,
                  Comment TEXT
              )
    ''' )
    db.commit()
    db.close()


if __name__ == '__main__':
    init_db()

with sqlite3.connect('data/database.db') as db:
    data = pd.read_excel('Ваш файл excel')
    data.to_sql('reviews_db', con=db, if_exists='replace', index=False)
