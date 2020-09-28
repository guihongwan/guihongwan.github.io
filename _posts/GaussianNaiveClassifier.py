import numpy as np
import pandas as pd
import scipy
import math

root = '../../'
import sys
sys.path.append(root+"pylib")

import tool

class NaiveClassifier:
	def __init__(self):
		pass

	def __prior(self):
		'''
		Calculate the probability for each class.
		@information used: self.y, self.n
		@ouput:self.priors
		'''

		self.priors = {}
		counts = self.y.value_counts().to_dict()
		for k, v in counts.items():
			self.priors[k] = v / self.y.size

	def __mean_variance(self):
		'''
		Calculate the mean, variance and so on for each class
		'''

		self.mean = {}
		self.variance = {}
		self.determinant = {}

		for c in self.y.unique():
			idxes = self.y==c
			X = self.X[idxes,:]

			# mean
			mu = np.mean(X,0).reshape((-1,1))
			self.mean[c] = mu
            
            # covariance
			Xc = X-mu.T
			n,m = Xc.shape
			# var = np.cov(Xc.T)
			var = (Xc.T@Xc)/(n-1)
			self.variance[c] = var
            
            # determinant
			self.determinant[c] = np.linalg.det(var)
			# deal with Singular matrix
			if np.linalg.det(var) <= 0:
				# tool.printred('nonpositive determinant!!! ' + str(np.linalg.det(var)))
				rank = np.linalg.matrix_rank(var)
				D, V = tool.EVD(var)
				D = D[:rank]
				determinant = 1
				for d in D:
					determinant = determinant*d
				self.determinant[c] = determinant

	def __calculate_Gaussian_probability(self, x, c):
		'''
		x: the test data point
		c: class
		'''
		u = self.mean[c]
		C = self.variance[c]
		determinant = self.determinant[c]
		
		x = x.reshape((-1,1))

		m = x.shape[0]
		part1 = ((math.sqrt(2*math.pi))**m)*(determinant**0.5)
		if part1 != 0:
		    part1 = 1/part1 # pay attention
        
		md = (x-u).T@np.linalg.pinv(C)@(x-u)

		part2 = (-1/2)*md
		part2 = math.e**part2

		return (part1*part2)[0,0]

	def fit(self, X, y):
	    self.X = X
	    self.y = pd.Series(y)
	    self.n = X.shape[0]
	    
	    self.__prior()
	    self.__mean_variance()

	def predict(self, X_test):
		n, m = X_test.shape
		y_pre = []
		for i in range(n):
			x_i = X_test[i,:].reshape((-1,1))
			P = {}
			for c in self.y.unique():
			    p = self.__calculate_Gaussian_probability(x_i, c)
			    p = p*self.priors[c]
			    P[c] = p
			P = tool.normalizeDict(P)
			y_pre.append(tool.argmaxDict(P))

		return y_pre

	def predict_proba(self, X_test):
		n, m = X_test.shape
		y_pre = []
		for i in range(n):
			x_i = X_test[i,:].reshape((-1,1))
			P = {}
			for c in self.y.unique():
				p = self.__calculate_Gaussian_probability(x_i, c)
				p = p*self.priors[c]
				P[c] = p
			P = tool.normalizeDict(P)
		return list(tool.sortDictbyKey(P).values())














