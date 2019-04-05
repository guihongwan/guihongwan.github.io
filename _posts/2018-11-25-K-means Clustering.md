---  
tag: Clustering
---

# Problem

Given n data points and k, group data into k clusters.    
$\textbf{Input:}$ n feature vectors $x_1, ..., x_n$ and an integer k.    
$\textbf{Output:}$ Partitions of the data into k classes.      


# K-means clustering

Define the quantization error by:     
$E = \sum\limits_{j=1}^k \sum\limits_{c(i)=j} |x_i - u_j|^2 = \sum\limits_{i=1}^n |x_i-u_{c(i)}|^2$     
$c(i)=j$ means $x_i$ belongs to class j    
$u_j$ is the mean of all points in class j. Similarly, $u_{c(i)}$ is the mean of class to which $x_i$ belongs.     


Keeping c(i) unchanged(fix the clusters), the value of $u_j$ that minimizes E: can be computed by taking the derivative of E with respect to $u_j$ and equating to 0.     

$\frac{\partial}{\partial u_j} \mid x_i - u_j \mid ^2 = \frac{\partial}{\partial u_j}(\mid x_i \mid^2 - 2x_i^Tu_j + \mid u_j \mid^2) = -2x_i + 2u_j = 2u_j - 2x_i$ 

Therefore:    
$\frac{\partial}{\partial u_j} \frac E 2 = \sum\limits_{c(i)=j}(u_j - u_i) = \sum\limits_{c(i)=j}u_j - \sum\limits_{c(i)=j}x_i = m_j u_j - \sum\limits_{c(i)=j} x_i$    
$m_j$ is the number of points in class j.    

$m_j u_j - \sum\limits_{c(i)} x_i = 0$    
Hence: $ {u_j = \frac{\sum\limits_{c(i)=j} x_i} {m_j} }$

Keeping $u_j$ unchanged, the value of c(i) that minimizes E is:      
for all i, $c(i) = arg\min\limits_j |x_i - u_j|^2 $

# Typical Algorithm
0. Start with a guess for $u_1,...,u_k$, or a guess for the c(i)
1. Iterate until convergence.

# Lloyd’s algorithm
0. Select the initial $u_i$ uniformly at random from $x_i, ..., x_n$.

# Implementation of Lloyd’s algorithm


```python
def randomInt(low, high, size):
    '''
    return a list with all values integers, no repeating, and with len equals to size.
    [low, high): the range
    '''
    import numpy as np
    ret = []
    while(len(ret) < size):
        v = np.random.randint(low, high)
        if v not in ret:
            ret.append(v)
    return ret

ret = randomInt(0, 5, 3)
print(ret)
```

    [2, 3, 4]



```python
def convertSoftToHardClustering(C, n, k):
    '''
    C: is a dictionary. {(0, 2): 0.9, ...} means x_0 belongs to class 2 with possibility 0.9
    n is number of points in data
    k is the number of Clusters
    '''
    import numpy as np
#     print(C)
    
    Labels = []
    for i in range(n):
        possible_class = {}
        for j in range(k):
            possible_class[j]= C.get((i,j), 0) 
            
#         print('possible_class for ', i, possible_class)
        
        import operator
        final_c = max(possible_class.items(), key=operator.itemgetter(1))[0]
        final_c += 1
        
        Labels.append(final_c)
    return Labels

# test
C = {(0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.3, (1, 1): 0.7, (2, 1): 1, (3, 1): 1}
n = 4
k = 2
L = convertSoftToHardClustering(C, n, k)
print(L)
```

    [1, 2, 2, 2]



```python
def KmeanLloyd(Xt, k, r):
    DEBUG = False
    import numpy as np
    
    n,m = Xt.shape # n is number of points
    C = {} #{(0, 0): 1, ...} means x_0 belongs to class 0
    
    for iteration in range(r):
        if DEBUG: print('----iteration:', iteration)
        # 1. U
        U = [] #[(5,1),...] means u_0 is (5,1)
        if (iteration == 0):# randomly
            # Select initial u_i
            
            # for test HW7 Q1
#             U.append((5,1))
#             U.append((6,0))

            # for test HW7 Q2
#             U.append((1,0))
#             U.append((4,2))
            
            idx = randomInt(0, n, k)
            for i in idx:
                U.append(Xt[i,:])
                
        else: # calulate U
            for j in range(k):
                sum_j = 0
                m_j = 0 # number of points in j
                for i in range(n):
                    if C.get((i,j), 0) == 1:
                        sum_j += Xt[i,:]
                        m_j += 1
                U.append(tuple(sum_j/m_j))
        if DEBUG: print('U:', U)
                
        
        # 2. C
        # 2.1 calculate distances |x_i - u_j|^2
        
        #[[26.0, 36.0], ...] the first one means the distance of x_0 to all u_j in U
        D = []
        for i in range(n):
            ds = []
            for u in U:
                d = sum((Xt[i,:] - u)**2)
                ds.append(d)
            D.append(ds)
#         print('D:', D)
        
        # 2.2 assign class
        C = {}
        for i in range(len(D)):
            j = np.argmin(D[i])
            C[i,j] = 1
#         if DEBUG: print('C:', C)
        
    # convert C into labels
    L = convertSoftToHardClustering(C, n, k)
    E = sum([min(d) for d in D])
    print('labels:', L)
    print('E:', E)        
    return L  
```

# Evaluation


```python
def plotData(Xt):
    '''
    Xt: a row is a point
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(Xt[:,0], Xt[:,1])
    ax.axis('equal')
def plotClusters(Xt, L):
    '''
    Xt: a row is a point
    L: labels coresponding to Xt
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.axis('equal')
    n,m = Xt.shape # n is number of points
    
    Group = []
    for j in range(k):
        group_j = []
        for i in range(n):
            if L[i] == (j+1):
                group_j.append(Xt[i,:])
        Group.append(group_j)
    for group in Group:
        g = np.array(group)
        ax.scatter(g[:,0], g[:,1])
```


```python
# Load Data
import numpy as np
filename = "simpledata.csv"
filename = "simpledata6.csv"
Xt = np.genfromtxt(filename, delimiter=',', autostrip=True)
print(Xt)
```

    [[ 6.  3.]
     [ 8.  0.]
     [ 4.  9.]
     [ 0.  0.]
     [ 1.  3.]
     [ 6.  5.]
     [ 5.  8.]]



```python
plotData(Xt)
```


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeans_0.png" width="300"/>




```python
k = 2
r = 5
L = KmeanLloyd(Xt, k, r)
plotClusters(Xt, L)
```

    labels: [1, 1, 2, 1, 1, 2, 2]
    E: 64.4166666667



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeans_2.png" width="300"/>


```python
k = 3
r = 5
C = KmeanLloyd(Xt, k, r)
plotClusters(Xt, C)
```

    labels: [1, 1, 3, 2, 2, 1, 3]
    E: 21.3333333333


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeans_1.png" width="300"/>



```python

```
