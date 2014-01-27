"""
Implementation of the KNNLearner class
"""

import numpy as np
import math

class KNNLearner:

	# k = Number of Nearest Neighbours, default value is 3
	def __init__(self, k = 3):
		self.k = k
		
	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		
	def query(self, Xtest):
		YPredicted = np.zeros(len(Xtest))
		
		# For Each Testing data, Calculate Y using the K-Nearest Neighbours
		for i in range(len(Xtest)):
			# Create an Array to hold the distances to all of the training data
			distances = np.zeros(len(self.Xtrain))
		
			# Calculate the distances for all of the training data points
			for j in range(len(self.Xtrain)):				
				for k in range(len(self.Xtrain[0])):
					
					distances[j] += math.pow(Xtest[i][k] - self.Xtrain[j,k], 2)
				
			
			# Get the array of sorted indices using argsort
			sorted_index_array = np.argsort(distances)

			# Get the K-nearest neighbours from the sorted array
			k_nearest_neighbours = self.Ytrain[sorted_index_array[0:self.k]]

			# Get the Y using the Mean of the K-Nearest Nieghbours	
			YPredicted[i] = np.mean(k_nearest_neighbours)
		
		return YPredicted	
