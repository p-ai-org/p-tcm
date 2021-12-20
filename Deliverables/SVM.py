import pandas as pd
from process_data import data_process
from sklearn.preprocessing import MinMaxScaler


from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import f1_score, accuracy_score

from sklearn.svm import SVC

def svm(df):
    X = df.iloc[:, 1:-1].drop(columns="Mode")
    y = df["Mode_code"].astype('int')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)
    
    # standardize and scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    rbf = SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    poly = SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)
    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

if __name__ == "__main__":
    df = pd.read_csv('./data/raw_data_cleaned.csv')
    df = data_process(df)
    df = df.drop(["Plain Occurences", "Cool Occurences", "Warm Occurences", "Cold Occurences", "Heavy Cold Occurences", "Heavy Warm Occurences","Hot Occurences", "Heavy Hot Occurences", '% Plain', '% Cool', '% Warm', '% Cold', '% Heavy Cold', '% Heavy Warm','% Hot', '% Heavy Hot', 'Plain', 'Cold', 'Hot', 'hot_cold_scale'], axis = 1)
    svm(df)

