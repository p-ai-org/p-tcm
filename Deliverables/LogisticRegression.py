import pandas as pd
from process_data import data_process
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression




def logistic_regression(df):
    """
    Apply sample logistic regression model to dataframe

    df : (dataframe)
    x : (compounds column as int)
    y : (classification column as int)
    """
    # values of each column
    x = df.iloc[:, 1:-1].drop(columns="Mode")
    y = df["Mode_code"].astype('int')
    # split data into training and test set
    # 75 training, 25 testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # standardize and scale data
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    # modeling + analysis
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
    model.fit(xtrain, ytrain.ravel())
    y_pred = model.predict(xtest)

    # confusion matrix of test size
    conf_m = confusion_matrix(ytest, y_pred)
    print("Confusion Matrix : ", conf_m)
    
    plot_confusion_matrix(model, xtest, ytest)

    # accuracy score of test size
    print ("Accuracy : ", accuracy_score(ytest, y_pred))

if __name__ == "__main__":
    df = pd.read_csv('./data/raw_data_cleaned.csv')
    df = data_process(df)
    df = df.drop(["Plain Occurences", "Cool Occurences", "Warm Occurences", "Cold Occurences", "Heavy Cold Occurences", "Heavy Warm Occurences","Hot Occurences", "Heavy Hot Occurences", '% Plain', '% Cool', '% Warm', '% Cold', '% Heavy Cold', '% Heavy Warm','% Hot', '% Heavy Hot', 'Plain', 'Cold', 'Hot', 'hot_cold_scale'], axis = 1)
    logistic_regression(df)