from sklearn import preprocessing
def label_encoder(df, column):
    # reading word labels
    label_encoder = preprocessing.LabelEncoder()

    # encode word labels in column
    df[column] = label_encoder.fit_transform(df[column])

    df[column].unique()

    return df