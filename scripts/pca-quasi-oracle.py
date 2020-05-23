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
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#update when necessary
DATA_FILE = "tedssd_puf_2017_training.csv.000"

#load dataset
df = pd.read_csv(DATA_FILE)
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

# Original code
# desiredChars = ["LOS", "AGE", "PSOURCE", "SUB1_D", "FRSTUSE1", "FREQ1", "EMPLOY", "ALCDRUG", "EDUC", "RACE", "GENDER"]
# indxsToDelete = []
# for key, value in dataFields.items():
# 	if key not in desiredChars:
# 		indxsToDelete.append(value)
#
# X = np.delete(dataset, indxsToDelete, 1)

#prepare data for PCA
print("Importing data")
data = pd.read_csv(DATA_FILE)
data.drop(data.iloc[:, 0:2], inplace=True, axis=1)
training = data.to_numpy()

# Kernel PCA
# Requires lots of space, very slow
# NUM_COMPONENTS = 50
# pca = KernelPCA(n_components=NUM_COMPONENTS, kernel="cosine")
# print("Applying PCA")
# X = pca.fit_transform(training)

# Normal PCA
# NUM_COMPONENTS = 11
# pca = PCA(n_components=NUM_COMPONENTS)
# print("Applying PCA")
# X = pca.fit_transform(training)

#Incremental PCA
NUM_COMPONENTS = 50
pca = IncrementalPCA(n_components=NUM_COMPONENTS, batch_size=1024)
print("Applying PCA")
X = pca.fit_transform(training)

print(X)
print(X.shape)
print(Y)
print(Y.shape)

def create_baseline():
	model = Sequential()
	model.add(Dense(NUM_COMPONENTS, input_dim=NUM_COMPONENTS, activation = 'relu'))
	model.add(Dense(9, activation = 'relu'))
	model.add(Dense(9, activation = 'relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
	return model

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=256, verbose=1)
kfold = StratifiedKFold(n_splits = 2, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
