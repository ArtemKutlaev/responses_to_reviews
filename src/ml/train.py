from sklearn.model_selection import train_test_split
from src.ml.prepare_data import get_prepared_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.config_path import MODEL_PATH,VECTORIZER_PATH

X,y,vectorizer,comments = get_prepared_data()

X_train,X_test,y_train,y_test,comments_train,comments_test = train_test_split(X,y,comments,test_size=0.3,random_state=1)
model = RandomForestClassifier(n_estimators=100,max_depth=100,random_state=1,n_jobs=-1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

joblib.dump(model,MODEL_PATH)
joblib.dump(vectorizer,VECTORIZER_PATH)


#Код снизу нужен для отладки, он выводит отзывы,которые классифицировались по какой-то причине неправильно
# label_names = {
#     0:'Негативный',
#     1:'Нейтральный',
#     2:'Позитивный'
# }
# y_test_list = list(y_test)
# comments_test_list = list(comments_test)

# errors_count = 0

# for comment,true_label, pred_label in zip(comments_test_list,y_test_list,y_pred):
#     if true_label != pred_label: 
#         errors_count+=1
#         print(f'Отзыв: {comment}')
#         print(f'Настоящий класс :{true_label}({label_names.get(true_label,true_label)})')
#         print(f'Предсказанный класс:{pred_label}({label_names.get(pred_label,pred_label)})')

# print(errors_count)       



