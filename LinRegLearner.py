"""
Implementation of the LinRegLearner class
"""
import numpy as np

class LinRegLearner:

	def __init__(self):
		self.X = None
		self.Y = None
		self.m = None
		self.c = None

	def addEvidence(self, X, Y):
		self.X = X
		self.Y = Y

		A = np.hstack([self.X, np.ones((len(self.X[:, 0]), 1))])
		m = np.linalg.lstsq(A, self.Y)[0]
		self.m = m[:-1]
		self.c = m[-1]
		
	def query(self, X_temp): 
		return X_temp.dot(self.m) + self.c
