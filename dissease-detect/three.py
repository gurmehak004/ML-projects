import pandas as pd
import numpy as np
import datetime
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC

data=pd.read_csv('dissease-detect/diseasefile.csv')
print(data.head())
print(data.isnull().sum())


data['age'] = data['age'] / 365.25

# Now that we have a better age column, we can drop the old ones
data=data.drop('date',axis=1)
data=data.drop('id',axis=1)
data=data.drop('country',axis=1)

print(data.head())

# Scaling numerical values, including the new 'age' column
print("\n handling the numerical values")
scaler=StandardScaler()
num_cols=['age', 'ap_hi','ap_lo','height','weight']
data[num_cols]=scaler.fit_transform(data[num_cols])
print(data.head())

# One-hot encoding on occupation
data=pd.get_dummies(data,columns=['occupation'],drop_first=True)
print(data.head())
data = data.astype(int)

X=data.drop('disease',axis=1)
y=data['disease']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Training the model 
print("Logistic regression model")
logreg=LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print("\n Decission tree model")
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print("\n KNN model")
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print("\n svm model")
svm_model=SVC(kernel='rbf',random_state=42)
svm_model.fit(X_train,y_train)
y_pred=svm_model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
