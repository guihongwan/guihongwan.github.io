---  
tag: Clustering
---

# Kmeans++ Algorithm
The only difference between kmeans++ and Lloyd's kmeans is the procedure of selecting the initial means.       
0.1 Select $u_1$ uniformly at random from $x_1,...,x_n$    
0.2 Assuming that $u_1,...,u_t$ are already selected, with t < k.        
Using the following procedure to select $u_{t+1}$    
- P = []    
for each $x_i$ compute the value $d_i$ as:    
$d_i = \min\limits_j\mid{x_i-u_j}\mid^2$    
P.append($d_i$)     
why min()? The possibility that $x_i$ is chosen as $u_{t+1}$ is proportional to $d_i$      
If $x_i$ is close to any of $u_1,...,u_t$, we hope the possibility is low.     
- normalize P
- Select one of the $x_i$ at random accordint to the probabilities $p_i$. Set it as $u_{t+1}$.


# Implementation


```python
def randomInt(low, high, size):
    '''
    return a list with all values are integers, no repeating, and with len equals to size.
    '''
    import numpy as np
    ret = []
    while(len(ret) < size):
        v = np.random.randint(low, high)
        if v not in ret:
            ret.append(v)
    return ret

def convertSoftToHardClustering(C, n, k):
    '''
    C: is a dictionary. {(0, 2): 0.9, ...} means x_0 belongs to class 2 with possibility 0.9
    n is number of points in data
    k is the number of Clusters
    '''
    import numpy as np
    
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

def normalizeP(P):
    '''
    P is a list
    '''
    Z = sum(P)
    P = P/Z
    return P
```


```python
def initalstepforLloyd(Xt, k):
    U = []
    n,m = Xt.shape # n is number of points
    idx = randomInt(0, n, k)
    for i in idx:
        U.append(Xt[i,:])
    return U
    
def initalstepforKmeansPP(Xt, k):
    '''
    Xt: a row is a point
    k: the number of clusters
    here, k means that we need to choose len(U) = k
    '''
    
    n,m = Xt.shape # n is number of points
    
    #Select u_0
    idx = randomInt(0, n, 1)
    U = [Xt[j,:] for j in idx]
    
    # Select u_1,...u_{k-1}
    for j in range(1,k):
        # calculate the P
        P = []
        for i in range(n):
            # calculate distance of each x_i to U
            ds = []
            for u in U:
                d = sum((Xt[i,:] - u)**2)
                ds.append(d)
            P.append(min(ds))
        P = normalizeP(P)
        #print(P)
        
        # select u_j
        idx_u_j = np.random.choice(n, 1, p=P)[0]
        U.append(Xt[idx_u_j,:])
            
    return U
```


```python
def Kmeans(Xt, k, r, initialFun):
    DEBUG = False
    import numpy as np
    
    n,m = Xt.shape # n is number of points
    C = {} #{(0, 0): 1, ...} means x_0 belongs to class 0
    
    for iteration in range(r):
        if DEBUG: print('----iteration:', iteration)
        # 1. U
        U = [] #[(5,1),...] means u_0 is (5,1)
        if (iteration == 0):
            # Select initial U
            U = initialFun(Xt, k)
                
        else: # calulate U
            for j in range(k):
                sum_j = 0
                m_j = 0
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
        if DEBUG: print('D:', D)
        
        # 2.2 assign class
        C = {}
        for i in range(len(D)):
            j = np.argmin(D[i])
            C[i,j] = 1
        if DEBUG: print('C:', C)
            
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
# for test
# Load Data
import numpy as np
filename = "simpledata.csv"
Xt = np.genfromtxt(filename, delimiter=',', autostrip=True)
k = 2
```


```python
plotData(Xt)
```


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeanspp_0.png" width="300"/>



```python
k = 2
r = 5
L = Kmeans(Xt, k, r, initialFun=initalstepforLloyd)
plotClusters(Xt, L)
```

    labels: [2, 1, 2, 1]
    E: 64.0



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeanspp_1.png" width="300"/>



```python
k = 2
r = 5
L = Kmeans(Xt, k, r, initialFun=initalstepforKmeansPP)
plotClusters(Xt, L)
```

    labels: [1, 1, 2, 2]
    E: 16.0



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kmeanspp_2.png" width="300"/>


# More
There is a weighted kmeans++ algorithm.     
Input: $x_1,...,x_n$ and associated weights $w_1,...,w_m$. k        
We can select $u_{t+1}$ at random from $x_1,...,x_n$ where the probability of selecting $x_i$ is proportional to $w_id_i$


<br>
For reproduction, please specify：[GHWAN's website](https://guihongwan.github.io) » [KMeans++ Clustering](https://guihongwan.github.io/2018/11/K-means++-Clustering/)
