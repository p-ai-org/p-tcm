import pandas as pd
from process_data import data_process
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn(df):
    x = df.iloc[:, 1:-1].drop(columns="Mode")
    y = df["Mode_code"].astype('int')
    
    xtrain, xtest, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # standardize and scale data
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(xtrain, y_train)

    y_pred = knn.predict(xtest)
    
    print ("Accuracy : ", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    df = pd.read_csv('./data/raw_data_cleaned.csv')
    df = data_process(df)
    df = df.drop(["Plain Occurences", "Cool Occurences", "Warm Occurences", "Cold Occurences", "Heavy Cold Occurences", "Heavy Warm Occurences","Hot Occurences", "Heavy Hot Occurences", '% Plain', '% Cool', '% Warm', '% Cold', '% Heavy Cold', '% Heavy Warm','% Hot', '% Heavy Hot', 'Plain', 'Cold', 'Hot', 'hot_cold_scale'], axis = 1)
    knn(df)