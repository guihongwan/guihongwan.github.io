---  
tag: Machine Learning 
---
# Description

The Gaussian density function of $m$-dimensional vectors is:    
$g(x;\mu,C) = {1\over (\sqrt{2\pi})^m |C|^{1/2}} e^{-{1 \over 2} (x-\mu)^TC^{-1}(x-\mu)}$    
where $\mu$ is the distribution mean, $C$ is the covaraince matrix. $|C|$ is the determinant of the matrix $C$.

The $\mu$ and $C$ can be estimated from the data.
$\mu = {\sum_{i=1}^n x_i \over m }$, 
$C = {\sum_{i=1}^n (x_i-\mu)(x_i-\mu)^T \over m-1 }$.

# Discriminant function

If $g(x;\mu_1,C_1)P(h_1) > g(x;\mu_2,C_2)P(h_2)$, then $x$ is classified as $C_1$.      
Problem: there may be no determinant of matrix $C$.     
Solution: $ (x-\mu_1)^TC_1^{-1}(x-\mu_1) + b < (x-\mu_2)^TC_2^{-1}(x-\mu_2)$, where $b$ is a threshold.

# Implementation


```python
import numpy as np
import pandas as pd
import scipy
import math


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
```


```python
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import KFold

import tool
import data

# read data
dataset_location = "Iris.csv"

X, y= data.read_csv(dataset_location, shuffle=False)
n, m = X.shape
print(X.shape)

k = 1 # reduced dimension

f = n # LEAVE ONE OUT
seed = -1
# split
if seed < 0:
    kf = KFold(n_splits = f, shuffle = True)
else:
    kf = KFold(n_splits = f, random_state = seed, shuffle = True)
idxesLists = kf.split(X)
splits = []
for trainidx, testindx in idxesLists:
    splits.append((trainidx, testindx))

DEBUG = True
if DEBUG:
    accs_imp = 0
    accs_imp_reduce = 0

    for trainidx, testindx in splits:
        X_train = X[trainidx,:]
        y_train = y[trainidx]
        X_test = X[testindx,:]
        y_test = y[testindx]

        Xt_train = X_train.T
        Xt_test = X_test.T
        
        #1.preprocessing
        # remove mean
        mean = np.mean(Xt_train,1).reshape(m,-1)
        Xt_train = Xt_train - mean
        Xt_test = Xt_test - mean
        X_train = Xt_train.T
        X_test  = Xt_test.T
        
        # PCA: dimension reduction
        D, V = tool.EVD(Xt_train@Xt_train.T)
        V = V[:,:k]
        Wt_train = V.T@Xt_train

        W_train = Wt_train.T
        Wt_test = V.T@Xt_test
        W_test = Wt_test.T
        
        #2. TEST
        # my implementation: without PCA
        clf = NaiveClassifier()
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)

        diff = y_pre - y_test
        acc = 1 - np.count_nonzero(diff)/len(y_test)
        accs_imp += acc
            
        # my implementation: with PCA
        clf = NaiveClassifier()
        clf.fit(W_train, y_train)
        y_pre = clf.predict(W_test)

        diff = y_pre - y_test
        acc = 1 - np.count_nonzero(diff)/len(y_test)
        accs_imp_reduce += acc

    print('accs_imp     :',accs_imp/f)
    print('accs_imp pca :',accs_imp_reduce/f)
```

    (150, 4)
    accs_imp     : 0.9733333333333334
    accs_imp pca : 0.9266666666666666



```python

```


```python

```
