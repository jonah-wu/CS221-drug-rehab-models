"""
Author: Nathan Marks
Date: May 10, 2020
This is a data cleaner to remove rows missing information and unneeded columns.
"""

import numpy as np
import pandas as pd


class DataCleaner():
	def __init__(self, csv):
		#init dataframe
		self.dataset = pd.read_csv(csv)


	def removeCols(self, colsToRemove):
		'''
		Remove columns from the dataset that aren't wanted.
		args:
		colsToRemove: a list of column names to remove.
		'''
		self.dataset.drop(columns=colsToRemove, inplace=True)


	def keepOnlyCols(self, colsToKeep):
		'''
		Keep only the specificed columns in the dataset.
		args:
		colsToKeep: a list of column names to keep.
		'''
		self.dataset = self.dataset[colsToKeep]


	def removeRowsMissingData(self, necessaryFields='all'):
		'''
		Remove rows from the dataset that contain a -9 in the specified fields. If
		no field is specified, all rows with a -9 will be removed.
		NOTE:
		-9 almost always means missing/unknown/not collected/invalid
		For field IDU: -9 means no substances reported
		For field DSMCRIT: -9 means missing/unknown/not collected/invalid OR deferred diagnosis
		args:
		necessaryFields: a list of field (column) names to keep
		'''
		#a list of row indices to delete
		rowsToDelete = []
		for i in range(len(self.dataset.index)):
			#iloc gets the row at index i
			if -9 in list(self.dataset.iloc[i]):
				rowsToDelete.append(i)

		self.dataset.drop(rowsToDelete, axis='rows', inplace=True)


	def saveDataset(self, filename):
		'''
		Save the dataset as a csv to current directory.
		args:
		filename: name for the file
		'''
		self.dataset.to_csv(filename, index=False)

def main():
    cleaner = DataCleaner('tedsd_puf_2017-training.csv')
    featureLables = list(pd.read_csv('TEDS-D_LOS_Features.csv', delimiter=','))
    cleaner.keepOnlyCols(featureLables)
    cleaner.removeRowsMissingData()
    cleaner.saveDataset('FullCleanFeatures.csv')

if __name__ == "__main__":
    main()

