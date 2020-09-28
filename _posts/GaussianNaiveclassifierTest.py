import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import KFold

root = '../../'
import sys
sys.path.append(root+"pylib")
import tool
import data

import entropyV21 as entropy
from GaussianNaiveClassifier import NaiveClassifier

# read data
dataset_location = root+"dataset/ionosphere_org_33.csv"
dataset_location = root+"dataset/Iris.csv"


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
        U, Wt_train, _, _ = entropy.qwPCA(Xt_train, k)

        W_train = Wt_train.T
        Wt_test = U.T@Xt_test
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
    



        
        