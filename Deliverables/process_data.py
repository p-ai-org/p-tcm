import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing


# OLD
def minmax_normalize(df):
    '''
    normalize after train test split now
    '''
    
    X = df.drop(['Food_name'],axis=1)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    X_array = scaler.transform(X)
    new_data = pd.DataFrame(X_array,columns = X.columns)
    new_data.insert(0, 'Food_name', df['Food_name'])
    return new_data

def label_encoder(df):
    '''
    no longer using this, as use hot/code mode instead
    '''

    # reading word labels
    label_encoder = preprocessing.LabelEncoder()

    # encode word labels in column
    df['hot_cold_scale'] = label_encoder.fit_transform(df['hot_cold_scale'])

    df['hot_cold_scale'].unique()

    return df
#


# Active
def occurrences(df):
    df["% Plain"] = df["Plain Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Cool"] = df["Cool Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Warm"] = df["Warm Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Cold"] = df["Cold Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Heavy Cold"] = df["Heavy Cold Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Heavy Warm"] = df["Heavy Warm Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Hot"] = df["Hot Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])
    df["% Heavy Hot"] = df["Heavy Hot Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

    return df

def linear_scale(df):
    df["hot_cold_scale"] = (df["% Plain"] * (3.0/6.0)) + (df["% Cool"] * (2.0/6.0)) + (df["% Warm"] * (4.0/6.0)) + (df["% Cold"] * (1.0/6.0)) + (df["% Heavy Cold"] * (0.0/6.0)) + (df["% Heavy Warm"] * (4.5/6.0)) + (df["% Hot"] * (5.0/6.0)) + (df["% Heavy Hot"] * (6.0/6.0))

    return df

def mode(df):
    df['Plain'] = df['Plain Occurences'].astype(int)
    df['Cold'] = df['Cool Occurences'].astype(int) + df['Cold Occurences'].astype(int) + df['Heavy Cold Occurences'].astype(int)
    df['Hot'] = df['Warm Occurences'].astype(int) + df['Heavy Warm Occurences'].astype(int) + df['Hot Occurences'].astype(int) + df['Heavy Hot Occurences'].astype(int)
    newDf = pd.DataFrame(df.loc[:, ['Plain', 'Cold', 'Hot']].idxmax(axis=1))
    df["Mode"] = newDf[0]
    
    def label_mode(row):
        if row['Mode'] == 'Plain':
            return 1
        elif row['Mode'] == 'Cold':
            return 0
        elif row['Mode'] == 'Hot':
            return 2
        else:
            return 3

    df['Mode_code'] = df.apply (lambda row: label_mode(row), axis=1)
    
    return df

def iterative_imputation(df):
    ii_imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=10, random_state=1121218)

    name = df.loc[:, "Food_name"]
    inputs = df.loc[:, 'Water (g)':'pH']
    outputs = df.loc[:, 'pH':].drop(columns='pH')
    
    new_cols = inputs.columns[inputs.isnull().mean() < 0.3]
    inputs = inputs[new_cols]
    
    newinputs = ii_imp.fit_transform(inputs)
    newinputs = pd.DataFrame(newinputs, columns = new_cols) 


    return pd.concat([name, newinputs, outputs], axis=1)


def data_process(df):
    df = df.rename(columns={"Unnamed: 0": "Food_name"})
    # df = minmax_normalize(df)
    df = occurrences(df)
    df = linear_scale(df)
    df = mode(df)
    
    df = iterative_imputation(df)
  
    df = label_encoder(df)
    
    return df

# df = pd.read_csv('./data/raw_data_cleaned.csv')
# df = data_process(df)
# df.to_csv('./data/data_processed.csv')

