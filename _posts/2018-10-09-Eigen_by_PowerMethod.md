---
layout: post
title: "My notes for html&css 2"
date: 2017-03-13 
description: "My notes for html&css 2"  

tag: Data Representation 
---

# The "Power" Method

The 'power' method to compute dominant eigenvectors/eigenvalues.
In many situations, we are only interested in a small number of eigenvectors associated with the largest eigenvalues.     
In order to compute the K top eigenvectors/eigenvalues, set k in the algorithem to K+P.

# Implementation


```python
import pandas as pd
from numpy import linalg as LA
import math
from tools import tool

import matplotlib.pyplot as plt
%pylab inline
```


```python
def firstEigenVectorByPowerMethod(B, interation = 3):
    '''
    B should be positive semidefinite and symmetric
    '''
    import numpy as np
    m = B.shape[0]
    v = tool.getRandomMatrix(m, 1)
    v = v/np.linalg.norm(v)
    
    for i in range(interation):
        v = B.dot(v)
        v = v/np.linalg.norm(v)
    return v
    
def kEigenByPowerMethod(B, k, interation = 4):
    '''
    B should be positive semidefinite and symmetric
    '''
    import numpy as np
    # step 1: get Q
    m = B.shape[0]
    Q = tool.getRandomMatrix(m, k)
    Q,R = np.linalg.qr(Q)
    for i in range(interation):
        Q = B.dot(Q)
        Q,R = np.linalg.qr(Q)
    
    # step 2: get Vw and Sw
    W = Q.T@B@Q
    Sw, Vw = LA.eigh(W)
    #print(Vw)
    # step 3:
    V = Q@Vw
    
    return Sw, V

def kEigenByPowerMethodBigData(filename, k, interation = 7):
    '''
    The big data is stored in the filename. One row is an instance.
    Access the data: interation+1 PASSES
    '''
    
    # step 0: initialize
    m = 0
    # get m
    with open(filename, buffering = 1) as f:#bytes 1MB
        for x in f:
            x = x.split()
            x_array = x[0].split(',')
            m = len(x_array)
            break
    
    Q = tool.getRandomMatrix(m, k)
    Q,R = np.linalg.qr(Q) # orthonormalize
    
    # step 1: get Q
    for i in range(interation):
        Q_m = np.zeros(m*k).reshape((m, k))
        with open(filename, buffering = 1024) as f: #1000000bytes 1MB
            for x in f:
                x = x.split() # remove \n
                x_array = x[0].split(',')
                x_list = [float(x) for x in x_array]
                x = numpy.array(x_list).reshape((m,1))
                t = Q.T@x
                y = x@t.T
                Q_m += y
            Q,R = np.linalg.qr(Q_m)
            
    #print(Q)
    
    # step 2: get Vw and Sw
    W = np.zeros(k*k).reshape((k, k))
    with open(filename, buffering = 1024) as f: #1000000bytes 1MB
        for x in f:
            x = x.split()
            x_array = x[0].split(',')
            x_list = [float(x) for x in x_array]
            x = np.array(x_list).reshape((m,1))
            t = Q.T@x
            item = t@t.T
            W += item
            
    Sw, Vw = LA.eigh(W)
    
    # step 3:
    V = Q@Vw
    
    return Sw, V
```

# Test


```python
X = np.array([
    [8,1,3],
    [1,2,3],
    [3,3,6]
]) 
B = X@X.T
```


```python
v = firstEigenVectorByPowerMethod(B)
print(v)
```

    [[0.73008889]
     [0.28938343]
     [0.61905367]]



```python
LA.eigh(B)
```




    (array([  0.12200217,  22.18809133, 119.6899065 ]),
     array([[-0.08423029, -0.67816319, -0.73006845],
            [-0.86088999,  0.41847145, -0.28939605],
            [ 0.50177055,  0.60413271, -0.61907187]]))




```python
S,V = kEigenByPowerMethod(B, 2)
print(S)
print(V)
```

    [ 22.18809133 119.6899065 ]
    [[-0.67816319  0.73006845]
     [ 0.41847145  0.28939605]
     [ 0.60413271  0.61907187]]



```python
## Big data
```


```python
filename = "bigdata_001_simple.csv"
S, V = kEigenByPowerMethodBigData(filename, 2)
print(S)
print(V)
```

    [  30.76954689 3662.09655857]
    [[-0.57168593  0.36063127]
     [ 0.33090181  0.36106449]
     [-0.047278    0.36942682]
     [-0.29735869  0.40870643]
     [-0.15097054  0.38734094]
     [ 0.65825975  0.35214324]
     [ 0.13009147  0.40252022]]



```python
X = np.loadtxt(open(filename), delimiter=",")
X = X.T
print(X.shape)
B = X@X.T
S,V = LA.eigh(B)
print(S)
print(V[:,len(V)-2:])
```

    (7, 1431)
    [   6.00882644    8.7988231    12.92971415   16.53282546   21.4076593
       30.7823316  3662.09655857]
    [[-0.55649858  0.36063127]
     [ 0.3408585   0.36106449]
     [-0.04347747  0.36942682]
     [-0.29994595  0.40870643]
     [-0.16288017  0.38734094]
     [ 0.66580225  0.35214324]
     [ 0.11155431  0.40252022]]



```python
filename = "bigdata_001.csv" # 55.8MB
S, V = kEigenByPowerMethodBigData(filename, 2)
print(S)
print(V)
```

    [   3614.51560614 1600472.01190097]
    [[-0.07562964  0.11627082]
     [ 0.14788757  0.12118491]
     [-0.05809577  0.10635662]
     [ 0.08725253  0.12621243]
     [ 0.08240343  0.11543932]
     [-0.00603038  0.12124585]
     [-0.10804814  0.12346103]
     [-0.25125089  0.11564231]
     [ 0.12132702  0.10802389]
     [-0.05194385  0.12032881]
     [ 0.06940225  0.12852377]
     [ 0.03270737  0.11383544]
     [ 0.01805468  0.11760276]
     [ 0.08373352  0.11633076]
     [-0.12000518  0.10626265]
     [ 0.2030823   0.11602552]
     [ 0.01526243  0.12228737]
     [-0.00995609  0.12873822]
     [-0.16309077  0.11490977]
     [ 0.03083004  0.12258349]
     [ 0.02374899  0.11346865]
     [-0.35513034  0.11389562]
     [-0.09100688  0.11195579]
     [ 0.11524915  0.1219414 ]
     [-0.09672321  0.12188879]
     [-0.01146133  0.11241096]
     [-0.01466111  0.11920285]
     [-0.05558218  0.12046978]
     [ 0.06041797  0.1228788 ]
     [-0.06536994  0.12394048]
     [ 0.08361177  0.12570319]
     [ 0.07141119  0.11960643]
     [-0.0004164   0.11514405]
     [ 0.19938429  0.11784554]
     [ 0.13801493  0.11612932]
     [ 0.09496374  0.12168276]
     [ 0.147242    0.12052921]
     [ 0.05363292  0.12643052]
     [-0.03697178  0.11547657]
     [-0.24730186  0.12503072]
     [ 0.03755471  0.1234218 ]
     [ 0.05055162  0.11384138]
     [-0.00467865  0.11634615]
     [-0.01281893  0.12241125]
     [-0.14410459  0.12119896]
     [ 0.03855677  0.12185701]
     [-0.02615617  0.11432594]
     [-0.09010006  0.12509628]
     [-0.02197426  0.12152795]
     [-0.10605085  0.12496809]
     [ 0.18789159  0.11707072]
     [-0.07088175  0.11612094]
     [-0.05026065  0.10891846]
     [ 0.06061671  0.12010598]
     [-0.01855557  0.1145353 ]
     [ 0.05635161  0.11268655]
     [ 0.19349086  0.12665512]
     [-0.02038037  0.11695747]
     [ 0.07624213  0.12156831]
     [ 0.02843508  0.12278295]
     [-0.00444195  0.12472785]
     [-0.11311783  0.12878722]
     [ 0.07503942  0.12068819]
     [-0.09844351  0.10994149]
     [-0.31513908  0.11825811]
     [ 0.18297963  0.13271202]
     [ 0.16489358  0.11622749]
     [ 0.15613655  0.12474071]
     [-0.04857006  0.12605308]
     [-0.27083863  0.12601349]]


# HW2


```python
Q = np.array([
    [1],
    [1],
    [1]
])

Q,R = np.linalg.qr(Q)
print(Q)
```

    [[-0.57735027]
     [-0.57735027]
     [-0.57735027]]



```python
def kEigenByPowerMethodBigData(k, interation = 7):
    
    # step 0: initialize
    m = 3
    n = 6*10**6
    
    X_partial = np.array([
                    [1, 1, 1, 1, 1, 1],
                    [1, 2, 1, 2, 1, 2],
                    [1, 2, 3, 1, 2, 3]])
    
    Q = np.array([
        [1],
        [1],
        [1]
    ])
    
    Q,R = np.linalg.qr(Q) # orthonormalize
    print('Q1 I0 orthonormalized Q\n:', Q)
    
    # step 1: get Q
    for i in range(interation):
        Q_m = np.zeros(m*k).reshape((m, k))
        for column in range(n):
            x = X_partial[:,column%6].reshape((m,1))
#             if column < 7:
#                 print(x)
            t = Q.T@x
            y = x@t.T
            Q_m += y
            if column == 0:
                print("1.1.1 the sum is \n", Q_m)
            if column == 5:
                print("1.1.2 the sum is \n", Q_m)
            
        Q,R = np.linalg.qr(Q_m)
    print("1.1.3 the sum is \n", Q_m)
    print("1.2 the Q:\n", Q)
    
    # step 2: get Vw and Sw
    W = np.zeros(k*k).reshape((k, k))
    for column in range(n):
        x = X_partial[:,column%6].reshape((m,1))
        t = Q.T@x
        item = t@t.T
        W += item
            
    Sw, Vw = LA.eigh(W)
    print('2 EigenValue:\n', Sw)
    print('2 EigenVector:\n', Vw)
    
    # step 3:
    V = Q@Vw
    
    return Sw, V
```


```python
S, V = kEigenByPowerMethodBigData(1, 1)
print('3 V after first iteration:\n', V)
```

    Q1 I0 orthonormalized Q
    : [[-0.57735027]
     [-0.57735027]
     [-0.57735027]]
    1.1.1 the sum is 
     [[-1.73205081]
     [-1.73205081]
     [-1.73205081]]
    1.1.2 the sum is 
     [[-15.58845727]
     [-24.24871131]
     [-33.48631561]]
    1.1.3 the sum is 
     [[-15588457.26836596]
     [-24248711.30600813]
     [-33486315.61253054]]
    [[-0.35279803]
     [-0.54879694]
     [-0.75786244]]
    2 EigenValue:
     [46221273.68869358]
    2 EigenVector:
     [[1.]]
    3 V after first iteration:
     [[-0.35279803]
     [-0.54879694]
     [-0.75786244]]

