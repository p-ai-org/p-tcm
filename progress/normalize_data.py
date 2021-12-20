from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
	X = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
	scaler = MinMaxScaler()
	scaler.fit(X)
	X_array = scaler.transform(X)
	new_data = pd.DataFrame(X_array,columns = X.columns)
 	new_data['food_names'] = df['Unnamed: 0.1']
  
return new_data
