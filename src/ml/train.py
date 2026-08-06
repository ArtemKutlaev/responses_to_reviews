from sklearn.model_selection import train_test_split
from src.ml.prepare_data import get_prepared_data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from catboost import CatBoostClassifier
import joblib
from src.config_path import MODEL_PATH,VECTORIZER_PATH
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV


def train_model(X_train,y_train) -> CatBoostClassifier:
    """Обучает CatBoostClassifier на тренировочных данных, предварительно подберая лучшие гиперпараметры с помощью RandomizedSearchCV """
    param_dist = {
        'iterations' : [200,300,500,700,800,900],
        'depth' : [4,6,8],
        'learning_rate' : [0.01,0.03,0.05,0.1,0.2,0.3],
        'l2_leaf_reg' : [1,3,5]
    }
    model = CatBoostClassifier(
        loss_function='MultiClass',
        random_seed=1,
        silent=True,
        task_type='GPU'
    )
    model_hpo = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter = 10,
        cv =3,
        scoring = 'f1_macro',
        n_jobs=1,
        verbose=2
    )
    
    model_hpo.fit(X_train,y_train)
    print(f'Лучшие параметры: {model_hpo.best_params_}')
    return model_hpo.best_estimator_

def evaluate_model(model, X_test,y_test) -> None: 
    """Функция,которая оценивает модель,выводит метрики и строит матрицу ошибок"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    report = classification_report(
    y_true=y_test,
    y_pred=y_pred,
    labels =[0,1,2],
    target_names=[
        "negative",
        "neutral",
        "positive"
    ],
    digits=4,
    zero_division=0
    )
    
    print(f'Accuracy_score:{accuracy}')
    print('\nClassification Report:\n',report)
    
    #Построение матрицы ошибок
    c_matrix = confusion_matrix(
    y_true = y_test,
    y_pred=y_pred,
    labels=[0,1,2]
    )
    disp = ConfusionMatrixDisplay(
    confusion_matrix=c_matrix,display_labels=["negative","neutral","positive"]
    )
    disp.plot(cmap='Blues')
    plt.title('Матрица ошибок по отзывам WB')
    plt.show()


if __name__ == '__main__':
    X,y,vectorizer = get_prepared_data()
    X_train,X_test,y_train,y_test= train_test_split(
        X,y,test_size=0.3,
        random_state=1,
        stratify=y
    )
    model = train_model(X_train,y_train)
    evaluate_model(model,X_test,y_test)
    joblib.dump(model,MODEL_PATH)
    joblib.dump(vectorizer,VECTORIZER_PATH)
    


