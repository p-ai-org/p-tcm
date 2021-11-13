from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def logistic_regression(df, x, y):
    """
    Apply sample logistic regression model to dataframe

    df : (dataframe)
    x : (compounds column as int)
    y : (classification column as int)
    """
    # values of each column
    x = df.iloc[1:, [x]].values
    y = df.iloc[1:, [y]].values

    # split data into training and test set
    # 75 training, 25 testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 25, random_state = 0)

    # standardize and scale data
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    # modeling + analysis
    model = LogisticRegression(random_state=0)
    model.fit(xtrain, ytrain.ravel())
    y_pred = model.predict(xtest)

    # confusion matrix of test size
    conf_m = confusion_matrix(ytest, y_pred)
    print("Confusion Matrix : ", conf_m)

    # accuracy score of test size
    print ("Accuracy : ", accuracy_score(ytest, y_pred))