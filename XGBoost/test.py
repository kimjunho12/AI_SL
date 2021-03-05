from xgboost import plot_importance
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

import pandas as pd

titanic = pd.read_csv('./data/titanic.csv')
titanic.info()

X, y = titanic.drop(columns=['survived', 'name', 'sex', 'ticket', 'cabin', 'embarked', 'boat', 'home.dest']), titanic['survived']
X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test, y_train, y_test

model = XGBClassifier()
model.fit(X_train, y_train)
model

import matplotlib.pyplot as pyplot
plot_importance(model)

y_pred = model.predict(X_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc

con_mat = confusion_matrix(y_true, y_pred)      # confusion_matrix()?
con_mat

report = classification_report(y_true, y_pred)
print(report)