import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
df1 = pd.read_excel (r'/Users/eyzhang28/Downloads/inputs_trans.xlsx')
X = df1.drop(["Plain Occurences", "Cool Occurences", "Warm Occurences", "Cold Occurences", "Heavy Cold Occurences", "Heavy Warm Occurences","Hot Occurences", "Heavy Hot Occurences", "SVM Training", "Temperature"], axis = 1)
Y = df1["SVM Training"]
factor = pd.factorize(df1['Temperature'])
df1.Temperature = factor[0]
definitions = factor[1]
print(definitions)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['Actual Temp'], colnames=['Predicted Temp']))
