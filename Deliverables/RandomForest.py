import pandas as pd
import numpy as np
from process_data import data_process
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier



def random_forest(df):
    X = df.iloc[:, 1:-1].drop(columns="Mode")
    Y = df["Mode_code"].astype('int')

    factor = pd.factorize(df['Mode'])
    df.Mode = factor[0]
    definitions = factor[1]
    print(definitions)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 21)
    
    # Normalization 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    reversefactor = dict(zip(range(3),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual Temp'], colnames=['Predicted Temp']))
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    df = pd.read_csv('./data/raw_data_cleaned.csv')
    df = data_process(df)
    df = df.drop(["Plain Occurences", "Cool Occurences", "Warm Occurences", "Cold Occurences", "Heavy Cold Occurences", "Heavy Warm Occurences","Hot Occurences", "Heavy Hot Occurences", '% Plain', '% Cool', '% Warm', '% Cold', '% Heavy Cold', '% Heavy Warm','% Hot', '% Heavy Hot', 'Plain', 'Cold', 'Hot', 'hot_cold_scale'], axis = 1)
    random_forest(df)