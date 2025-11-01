import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import io

df = pd.read_csv('parkinsons.data')
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.columns())

fig = plt.figure(figsize=(10,6))
plt.hist(X=df.status)
plt.xlabel = 'status'
plt.ylabel = 'Frequency'


fig = plt.figure(figsize=(10,6))
sns.barplot(X=df.status,y=df.NHR)


fig = plt.figure(figsize=(10,6))
sns.barplot(X=df.status,y=df.HNR)

rows = 3
cols = 7
fig,ax = plt.subplots(nrows = rows,ncols= cols,figsize = (16,4))
col = df.columns
index = 1

for i in range(rows):
    for j in range(cols):
        sns.displot(df[col[index]],ax=ax[i][j])
        index = index+1

    plt.tight_layout


df.drop(['name'],axis=1,inplace=true)

X = df.drop(labels=['status'],axis=1)
y = df['status']
X.head()

y.head()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


log_reg = LogisticRegression.fit(X_train,y_train)

train_pred = log_reg.predict(X_train)
print('Accuracy on training data',accuracy_score(train_pred,y_train))

test_pred = log_reg.predict(X_test)
print('Accuracy on testing data',accuracy_score(test_pred,y_test))

print('Confusion matrix on training set is',confusion_matrix(y_train,train_pred))
print('Confusion matrix on testing set is',confusion_matrix(y_train,test_pred))


rf = RandomForestClassifier.fit(X_train,y_train)

train_pred1 = rf.predict(X_train)
print('Accuracy on training data',accuracy_score(train_pred,y_train))

test_pred1 = rf.predict(X_test)
print('Accuracy on testing data',accuracy_score(test_pred,y_test))

print('Confusion matrix on training set is',confusion_matrix(y_train,train_pred))
print('Confusion matrix on testing set is',confusion_matrix(y_train,test_pred))


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,cohen_kappa_score,recall_score,confusion_matrix


dt = DecisionTreeClassifier.fit(X_train,y_train)

train_pred2 = rf.predict(X_train)
print('Accuracy on training data',accuracy_score(train_pred2,y_train))

test_pred2 = rf.predict(X_test)
print('Accuracy on testing data',accuracy_score(test_pred2,y_test))

print('Confusion matrix on training set is',confusion_matrix(y_train,train_pred2))
print('Confusion matrix on testing set is',confusion_matrix(y_train,test_pred2))

nb = GaussianNB.fit(X_train,y_train)

train_pred4 = rf.predict(X_train)
print('Accuracy on training data',accuracy_score(train_pred4,y_train))

test_pred4 = rf.predict(X_test)
print('Accuracy on testing data',accuracy_score(test_pred4,y_test))

print('Confusion matrix on training set is',confusion_matrix(y_train,train_pred4))
print('Confusion matrix on testing set is',confusion_matrix(y_train,test_pred4))

