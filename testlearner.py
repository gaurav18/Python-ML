"""
Main file
"""

#Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import KNNLearner as KNN
import LinRegLearner as LRL
import time
from pylab import *
#import plot3ddata as plotting


def KNNLearner_main(filename):
	
	# Read data
	#data = np.genfromtxt(filename, delimiter = ',')
	data = np.loadtxt(filename,delimiter=',',skiprows=0)
	
	# Put data into separate arrays
	Xdata = data[:, 0:2]
	Ydata = data[:, 2]
	num_rows = len(data[:, 0])

	# First 60%: Training Data
	Xtrain = Xdata[0:0.6 * num_rows, :]
	Ytrain = Ydata[0:0.6 * num_rows]

	# Remaining 40%: Testing Data
	Xtest = Xdata[0.6 * num_rows:, :]
	Ytest = Ydata[0.6 * num_rows:]
	
	# Create a KNNLearner Objects for K = 3
	learner = KNN.KNNLearner(3)

	# Calculation of Average Training Time
	training_start_time = time.clock()
	learner.addEvidence(Xtrain, Ytrain)
	training_end_time = time.clock()
	total_train_time = training_end_time - training_start_time
	average_train_time = total_train_time / len(Xtrain[:, 0])
	
	# Printing the value
	print 'Running KNNLearner for data', filename
	print 'K Nearest Neighbours, K = 3:'
	print 'Average Training Time per instance = ', average_train_time
	
	# Average Testing Time Calculation
	start_query_time = time.clock()
	Ypredicted = learner.query(Xtest)
	end_query_time = time.clock()
	total_query_time = end_query_time - start_query_time
	average_query_time = total_query_time / len(Xtest[:, 0])
	
	# Prnting the value
	print 'Average Query Time per instance = ', average_query_time
	
	# Calculate Correlation Coefficient for the Actual and Predicted Data
	corr_coefficient = np.corrcoef(Ytest, Ypredicted)[0,1]
	print 'Correlation Coefficient between Actual and Predicted = ', corr_coefficient

	# Calculate the Root Mean Squared Error(RMS)
	rms = np.sqrt(np.mean(np.subtract(Ytest, Ypredicted)**2))
	print 'Root Mean Squared Error(RMS) = ' , rms
	print '\n'

	# Plot the Scatter plot for Expected Y vs Predicted Y
	scatter_name = 'KNN_ScatterPlot' + filename + '.pdf'
	plt.clf()
	plt.cla()
	plt.scatter(Ytest, Ypredicted, c ='r')
	plt.xlabel('Expected Y')
	plt.ylabel('Predicted Y')
	plt.savefig(scatter_name, format='pdf')

	# Create an array to hold all the correlation Coefficients
	corr_coefficients = np.zeros(50)
	rms_array = np.zeros(50)


	# Find k which give least rms
	for k in range(1,51):
		learner_k = KNN.KNNLearner(k)
		learner_k.addEvidence(Xtrain, Ytrain)
		Ypredicted_k = learner_k.query(Xtest)
		corr_coefficients[k - 1]= np.corrcoef(Ytest, Ypredicted_k)[0, 1]
		rms_array[k - 1] = np.sqrt(np.mean(np.subtract(Ytest, Ypredicted_k)**2))

	# Plot the Graph of Correlation Coefficent Vs K
	plt.clf()
	plt.cla()
	plt.plot(range(1,51), corr_coefficients, c = "r")
	plt.xlabel('K')
	plt.ylabel('Correlation Coefficients')
	graphname = 'CorrCoeff Vs K for' + filename + '.pdf'
	plt.savefig(graphname, format = 'pdf')
	max_k = np.argmax(corr_coefficients) + 1
	print 'Best K = ', max_k
	print 'Correlation Coefficient for the best K = ', corr_coefficients[max_k - 1]
	print 'RMS Error for the Best K = ', rms_array[max_k - 1]
	print '\n'


def LinRegLearner_main(filename):
	
	# Read data
	data = np.genfromtxt(filename, delimiter = ',')
	
	# Put data into separate arrays
	Xdata = data[:, 0:2]
	Ydata = data[:, 2]
	num_rows = len(data[:, 0])

	# First 60%: Training Data
	Xtrain = Xdata[0:0.6 * num_rows, :]
	Ytrain = Ydata[0:0.6 * num_rows]

	# Remaining 40%: Testing Data
	Xtest = Xdata[0.6 * num_rows:, :]
	Ytest = Ydata[0.6 * num_rows:]
	
	# Create a Linear Regression Learner and calculate the avg training time
	linear_learner = LRL.LinRegLearner()
	training_start_time = time.clock()
	linear_learner.addEvidence(Xtrain, Ytrain)
	training_end_time = time.clock()
	total_train_time = training_end_time - training_start_time
	average_train_time = total_train_time / len(Xtrain[:, 0])
	
	# Printing the values
	print 'Running Linear Regression for data', filename
	print 'Linear Regression:'
	print 'Average Training Time per instance = ', average_train_time
	
	# Average Testing Time Calculation
	start_query_time = time.clock()
	Ypredicted = linear_learner.query(Xtest)
	end_query_time = time.clock()
	total_query_time = end_query_time - start_query_time
	average_query_time = total_query_time / len(Xtest[:, 0])
	
	# Printing the value
	print 'Average Query Time per instance = ', average_query_time
	
	# Calculate Correlation Coefficient for the Actual and Predicted Data
	corr_coefficient= np.corrcoef(Ytest, Ypredicted)[0,1]
	print 'Correlation Coefficient between Actual and Predicted = ', corr_coefficient

	# Calculate the Root Mean Squared Error(RMS)
	rms = np.sqrt(np.mean(np.subtract(Ytest, Ypredicted)**2))
	print 'Root Mean Squared Error(RMS) = ' , rms
	print '\n'

print '\n'
KNNLearner_main('data-classification-prob.csv')
KNNLearner_main('data-ripple-prob.csv')
#plotting.plot_data('data-classification-prob.csv')
LinRegLearner_main('data-classification-prob.csv')
LinRegLearner_main('data-ripple-prob.csv')
