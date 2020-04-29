import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#load dataset
df = pd.read_csv("../data/tedsd_2016_puf.csv")
dataset = df.to_numpy()
print(dataset)

#Extract Features
dataFields = {}
for x in range(len(df.columns)):
	dataFields.update({df.columns[x]: x})
print(dataFields)
indx = dataFields["REASON"]

Y = dataset[:,[indx]]
# Set to binary classification problem
print(Y)
Y = np.select([(Y == 1)], [1] , default = 0)
print(Y)
print(Y.shape)

print(dataset.shape)


desiredChars = ["LOS", "AGE", "PSOURCE", "SUB1_D", "FRSTUSE1", "FREQ1", "EMPLOY", "ALCDRUG", "EDUC", "RACE", "GENDER"]
indxsToDelete = []
for key, value in dataFields.items():
	if key not in desiredChars:
		indxsToDelete.append(value)

X = np.delete(dataset, indxsToDelete, 1)
print(X)
print(X.shape)
print(Y)
print(Y.shape)

def create_baseline():
	model = Sequential()
	model.add(Dense(len(desiredChars), input_dim=len(desiredChars), activation = 'relu'))
	model.add(Dense(9, activation = 'relu'))
	model.add(Dense(9, activation = 'relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
	return model

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=256, verbose=1)
kfold = StratifiedKFold(n_splits = 2, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




