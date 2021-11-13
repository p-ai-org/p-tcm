from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd


# dataset
df = pd.read_excel('/Users/shreya/rawData_occurrences.xlsx')
df.head()

df['Linear Scale'] = df['Linear Scale'] * 10
df['Linear Scale'] = df['Linear Scale'].fillna(0.0).astype(int)
y = df['Linear Scale'].to_numpy()
# print(y)


df.drop(columns=['Unnamed: 0.1'], inplace=True)
df.drop(columns=['Plain Occurences'], inplace=True)
df.drop(columns=['Cool Occurences'], inplace=True)
df.drop(columns=['Warm Occurences'], inplace=True)
df.drop(columns=['Cold Occurences'], inplace=True)
df.drop(columns=['Heavy Cold Occurences'], inplace=True)
df.drop(columns=['Heavy Warm Occurences'], inplace=True)
df.drop(columns=['Hot Occurences'], inplace=True)
df.drop(columns=['Heavy Hot Occurences'], inplace=True)
df.drop(columns=['% Plain'], inplace=True)
df.drop(columns=['% Cool'], inplace=True)
df.drop(columns=['% Warm'], inplace=True)
df.drop(columns=['% Cold'], inplace=True)
df.drop(columns=['% Heavy Cold'], inplace=True)
df.drop(columns=['% Heavy Warm'], inplace=True)
df.drop(columns=['% Hot'], inplace=True)
df.drop(columns=['% Heavy Hot'], inplace=True)
df.drop(columns=['Linear Scale'], inplace=True)


X = df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(
# fit a k-nearest neighbor model to the data
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
# divide by 10?
# print(knn)

df.to_excel('./knn_data.xlsx')
