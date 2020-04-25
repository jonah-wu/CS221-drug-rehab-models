"""
Author: Nathan Marks
Date: 4/24/20
This is a pretty simple linear regression model predicting the length of stay of individuals
in TEDS-D.
Dependencies: tedsd_puf_2017-training.csv, tedsd_puf_2017-testing.csv, TEDS-D_LOS_Features.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

print ('reading data')
teds_training = pd.read_csv('tedsd_puf_2017-training.csv')
teds_testing = pd.read_csv('tedsd_puf_2017-testing.csv')
#print (str(teds.head()))

print ('getting feature labels')
featureLables = list(pd.read_csv('TEDS-D_LOS_Features.csv', delimiter=','))
#print (str(featureLables))

#Length of Stay
y_label = "LOS"
print ('setting y')
yTrain = teds_training[y_label]
yTest = teds_testing[y_label]
#print (str(yTrain))

print ('setting x')
xTrain = teds_training[featureLables]
xTest = teds_testing[featureLables]
#print (str(xTrain.head()))

print('training')
lm = LinearRegression()
model = lm.fit(xTrain,yTrain)

print('predicting')
predictions = lm.predict(xTest)
#print (str(predictions))

print ('evaluating')
differences = np.array(yTest) - np.array(predictions)
totalLoss = (1/len(differences)) * (np.dot(differences, differences))
averageDifference = (1/len(differences)) * sum(abs(differences))
print ('averageDifference: ' + str(averageDifference))
print ('totalLoss: ' + str(totalLoss))
"""
Write predictions and actual y values to files
f = open("predictions.txt", "w")
for prediction in predictions:
    f.write(str(prediction) + '\n')
f = open("yTest.txt", "w")
for y in yTest:
    f.write(str(y) + '\n')
f.close()
"""
















