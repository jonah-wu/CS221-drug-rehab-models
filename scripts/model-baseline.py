import numpy as np 
import pandas as pd 

# Dataset Stats # 
teds = pd.read_csv("./data/TEDS_D.csv")
print(teds.head())

dataFields = []
numFields = 0
for col in teds.columns:
	dataFields.append(col)
	numFields += 1

y_label = "REASON"


# Our Predictor
successConditions = [
	(teds["LIVARAG"] == 3) & #Lived indptly at admission
	(teds["SUB1"] == 2) | (teds["SUB1"] == 4) & #The individual is in for either alcohol/pot (Assumption: less severe)
	(teds["EDUC"] >= 3) & #Individual has at least GED level of educ.
	(teds["EMPLOY"] == 1) | (teds["EMPLOY"] == 2) & # Has part/full time job.
	(teds["PSOURCE"] < 5) #Treatment referral source other than community,criminaljusti/missing
]
conditionLabels = [1]

# Prediction
teds["prediction"] = np.select(successConditions, conditionLabels, default=0)
print(teds["prediction"])


numExamples = len(teds.index)
assert(numExamples == len(teds["prediction"]))
predictedSuccess = np.sum(np.asarray(teds["prediction"])) / len(teds["prediction"])

print(str(predictedSuccess) + " percent of the individuals in this dataset were predicted to successfully complete treatment")
print("40 percent actually did based on the " + y_label + " field.")


# Calculate error
classificationList = ["TN", "TP","FN","FP"] #True Negative, True Positive, False Negative, False Positive
classificationConditions = [
	(teds["prediction"] == 0) & (teds[y_label] != 1), # 1 means Treatment Completed.
	(teds["prediction"] == 1) & (teds[y_label] == 1),
	(teds["prediction"] == 0) & (teds[y_label] == 1),
	(teds["prediction"] == 1) & (teds[y_label] != 1)
]


# Assign default labels (TN, TP, FN, FP to each of our examples based on our prediction and the y_label)
teds["classification"] = np.select(classificationConditions, classificationList, default= -1)
assert(numExamples == len(teds["classification"]))


print(teds["classification"].value_counts())
trueNeg = len(teds[teds['classification'] == "TN"])
truePos = len(teds[teds['classification'] == "TP"])
falseNeg = len(teds[teds['classification'] == "FN"])
falsePos = len(teds[teds['classification'] == "FP"])
assert(truePos + trueNeg + falsePos + falseNeg == numExamples)


# Calculate our accuracy/precision/recall/f1Score
model_accuracy =  (trueNeg + truePos) / numExamples
model_precision = truePos / (truePos + falsePos)
model_recall = truePos / (truePos + falseNeg)
model_f1Score = 2 * (model_precision * model_recall) / (model_precision + model_recall)


print("Model accuracy is " + str(model_accuracy))
print("Model precision is " + str(model_precision))
print("Model recall is " + str(model_recall))
print("Model f1 Score is " + str(model_f1Score))









