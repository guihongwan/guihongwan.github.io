---  
tag: Clustering
---

# Algorithm

Previously, the membership function c(i,j):    
c(i,j) = 1, if $x_i$ belongs to class j;    
c(i,j) = 0, if $x_i$ does not to class j.     

Here, we allow 0 <= c(i,j) <= 1, for a probabilistic description.     
The error criterion generalizes the k-means error criterion:      
$J = \sum\limits_{i=1}^n \sum\limits_{j=1}^k c(i,j)^m||x_i-u_j||^2$     
$E = \sum\limits_{i=1}^n |x_i-u_{c(i)}|^2$ (k-means)

m is the hyperparameter that controls how fuzzy the cluster will be.    
The higher it is, the fuzzier the cluster will be in the end.    

Given $u_j$, consider the membership values of $x_i$. It should be inversely related to $||x_i- u_j||$    
for all i,j, $\widehat{c}(i,j) = \frac 1 {||x_i- u_j||^{\frac 2 {m-1}}} $      
for all i,j, normalize $\widehat{c}(i,j)$ over $z_i$:         
$c(i,j) = \frac {\widehat{c}(i,j)} {z_i} $ where $z_i = \sum \limits_{j=1}^k \widehat{c}(i,j)$



Given the c(i,j), the new c-means are computed by:     
for all j:
$u_j = \frac{\sum\limits_{i=1}^n c(i,j)^mx_i} {\sum\limits_{i=1}^n c(i,j)^m}$

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
    posibilities = [item/Z for item in P]
    return posibilities
```


```python
def initalstepforLloyd(Xt, k):
    U = []
    n,m = Xt.shape # n is number of points
    idx = randomInt(0, n, k)
    for i in idx:
        U.append(Xt[i,:])
    return U
    
def initalstepforKmeansPP(Xt, k, W = None):
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
            posi = min(ds)
            if(W != None):
                posi = W[i]*posi
            P.append(posi )
        P = normalizeP(P)
        #print(P)
        
        # select u_j
        idx_u_j = np.random.choice(n, 1, p=P)[0]
        U.append(Xt[idx_u_j,:])
            
    return U

def initalstepforKmeansParallelFirst(Xt, p, li):
    '''
    Xt: a row is a point
    p: number of passes
    li: a value for the oversampling factor.
    '''
    
    n,m = Xt.shape # n is number of points
    
    #Select u_0
    idx = randomInt(0, n, 1)
    U = [Xt[j,:] for j in idx]
    
    # p passes
    for j in range(p):
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
        
        # select li points
        indexes = np.random.choice(n, li, p=P)
        for idx in indexes:
            U.append(Xt[idx,:])
    return U

def initalstepforKmeansParallelSecond(Xt, U1, k):
    '''
    U1: is obtained from the first step with len(U) >= k
    We use weighted KmeansPP to select k from U1
    
    COUNTS: the number of points that are cloasest to u_j.
            We can obtain this by Kmeans given U with one iteration.
    '''
    import numpy as np
    L,U,E = Kmeans(Xt, len(U1), 1, givenU=U1)
    COUNTS = []
    for j in range(1,len(U1)+1):
        COUNTS.append(L.count(j))
    W = normalizeP(COUNTS)
    
    U1 = np.array(U1)
    U = initalstepforKmeansPP(U1, k, W)
    return U

def initalstepforKmeansParallel(Xt, k):
    p = 5
    li = int(k/2)
    U1 = initalstepforKmeansParallelFirst(Xt, p, li)
    U = initalstepforKmeansParallelSecond(Xt, U1, k)
#     print('final U:', U)
    return U
    
```


```python
def Kmeans(Xt, k, r, initialFun=None, givenU=None):
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
            if(givenU == None):
                U = initialFun(Xt, k)
            else:
                U = givenU
                
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
    return (L,U,E)
```


```python
def Cmeans(Xt, k, r, mm, initialFun=None, givenU=None):
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
            if(givenU == None):
                U = initialFun(Xt, k)
            else:
                U = givenU
                
        else: # calulate U
            for j in range(k):# u_j
                sum_j = 0
                m_j = 0
                for i in range(n):
                    sum_j += (C.get((i,j), 0)**mm)*Xt[i,:]
                    m_j += C.get((i,j), 0)**mm
                U.append(tuple(sum_j/m_j))
#         if DEBUG: 
#         print('U:', U)
                
        
        # 2. C
        # 2.1 calculate distances |x_i - u_j|^2
        
        #[[26.0, 36.0], ...] the first one means the distance of x_0 to all u_j in U
        D = {} # here, D is C^.
        for i in range(n):
            for j in range(k):
                import math
                d = sum((Xt[i,:] - U[j])**2)
                if(d == 0):
                    D[i,j] = -1 # mark, in this case, C[i,j] = 1
                else:
                    D[i,j] = (1/d)**(1/(mm-1))
                
        if DEBUG: print('D:', D)
            
        #for D[i,j]==-1
        for i in range(n):
            for j in range(k):
                if(D[i,j]==-1):
                    for idx in range(k):
                        if(idx != j):
                            D[i,idx] = 0
                        else:
                            D[i,idx] = 1
        if DEBUG: print('D^:', D)
        
        # 2.2 assign class/normalize C^
        C = {}
        for i in range(n):
            z_i = 0
            for j in range(k):
                z_i += D[i,j]
            for j in range(k):
                C[i,j] = D[i,j]/z_i
        if DEBUG: print('C:', C)
            
    # convert C into labels
#     print('C', C)
    L = convertSoftToHardClustering(C, n, k)

    DD = []
    for i in range(n):
        ds = []
        for j in range(k):
            import math
#             d = math.sqrt(sum((Xt[i,:] - U[j])**2))
            d = sum((Xt[i,:] - U[j])**2)
            ds.append(d)
        dd = min(ds)
        DD.append(dd)
    E = sum(DD)
    
    return (L,U, E)
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
def plotClusters(Xt, L, U=None):
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
        nn= g.shape[0]
        if(nn > 0):
            ax.scatter(g[:,0], g[:,1], alpha=0.7)
    if (U!=None):
        U = np.array(U)
        ax.scatter(U[:,0], U[:,1], marker='X', s=70, c='r')
```


```python
# for test
# Load Data
import numpy as np
filename = "iris.data.txt"
Xt = np.genfromtxt(filename, delimiter=',', autostrip=True)
```


```python
k = 3
r = 20 # for converge, in the future, I will remove this. Simply, repeat until converge
Iterations = 10 # choose the best one. For future evaluation.
```


```python
#Kmeans
Es = []
Us = []
Ls = []
for i in range(Iterations):
    L,U,E = Kmeans(Xt, k, r, initialFun=initalstepforLloyd)
    Ls.append(L)
    Us.append(U)
    Es.append(E)
best = np.argmin(Es)
L = Ls[best]
U = Us[best]
E = Es[best]
for j in range(1,k+1):
    print(j,':', L.count(j))
print(E)
plotClusters(Xt, L, U)
```

    1 : 62
    2 : 50
    3 : 38
    78.9408414261



<!-- ![png](output_13_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/cmeans_0.png" width="300"/>


```python
#Kmeans++
Es = []
Us = []
Ls = []
for i in range(Iterations):
    L,U,E = Kmeans(Xt, k, r, initialFun=initalstepforKmeansPP)
    Ls.append(L)
    Us.append(U)
    Es.append(E)
best = np.argmin(Es)
L = Ls[best]
U = Us[best]
E = Es[best]
for j in range(1,k+1):
    print(j,':', L.count(j))
print(E)
print('U', U)
plotClusters(Xt, L, U)
```

    1 : 62
    2 : 50
    3 : 38
    78.9408414261
    U [(5.9016129032258071, 2.7483870967741941, 4.3935483870967751, 1.4338709677419357), (5.0059999999999993, 3.4180000000000006, 1.464, 0.24399999999999991), (6.8500000000000005, 3.0736842105263151, 5.7421052631578933, 2.0710526315789473)]



<!-- ![png](output_14_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/cmeans_1.png" width="300"/>



```python
#Kmeans Parallel
Es = []
Us = []
Ls = []
for i in range(Iterations):
    L,U,E = Kmeans(Xt, k, r, initialFun=initalstepforKmeansParallel)
    Ls.append(L)
    Us.append(U)
    Es.append(E)
best = np.argmin(Es)
L = Ls[best]
U = Us[best]
E = Es[best]
for j in range(1,k+1):
    print(j,':', L.count(j))
print(E)
print('U', U)
plotClusters(Xt, L, U)
```

    1 : 62
    2 : 50
    3 : 38
    78.9408414261
    U [(5.9016129032258071, 2.7483870967741941, 4.3935483870967751, 1.4338709677419357), (5.0059999999999993, 3.4180000000000006, 1.464, 0.24399999999999991), (6.8500000000000005, 3.0736842105263151, 5.7421052631578933, 2.0710526315789473)]



<!-- ![png](output_15_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/cmeans_2.png" width="300"/>


```python
#Cmeans
Es = []
Us = []
Ls = []
Iterations = 100
mm = 3
for i in range(Iterations):
    L,U,E = Cmeans(Xt, k, r, mm, initialFun=initalstepforLloyd)
    Ls.append(L)
    Us.append(U)
    Es.append(E)
best = np.argmin(Es)
L = Ls[best]
U = Us[best]
E = Es[best]
for j in range(1,k+1):
    print(j,':', L.count(j))
print(E)
print('U', U)
plotClusters(Xt, L, U)
```

    1 : 59
    2 : 41
    3 : 50
    80.97541316
    U [(5.9108592005116769, 2.7917436493396055, 4.3795234999340673, 1.3969150114682978), (6.695439882813317, 3.0376285801395135, 5.5518773819112255, 2.0356630177264541), (5.0010566452105811, 3.3893281471514598, 1.4942916006649354, 0.25196092517731689)]



<!-- ![png](output_16_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/cmeans_3.png" width="300"/>


```python
#Cmeans
Es = []
Us = []
Ls = []
Iterations = 100
mm = 0.3
for i in range(Iterations):
    L,U,E = Cmeans(Xt, k, r, mm, initialFun=initalstepforLloyd)
    Ls.append(L)
    Us.append(U)
    Es.append(E)
best = np.argmin(Es)
L = Ls[best]
U = Us[best]
E = Es[best]
for j in range(1,k+1):
    print(j,':', L.count(j))
print(E)
print('U', U)
plotClusters(Xt, L, U)
```

    1 : 0
    2 : 92
    3 : 58
    680.165925313
    U [(5.8435799406544042, 3.0539119304103188, 3.7593304117474982, 1.1989357046657578), (5.8428172563321166, 3.054184542678152, 3.7572771469913873, 1.1981035040438766), (5.8436021692224127, 3.053903968367373, 3.759390280759455, 1.1989599678164726)]



<!-- ![png](output_17_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/cmeans_4.png" width="300"/>



<br>
For reproduction, please specify：[GHWAN's website](https://guihongwan.github.io) » [Fuzzy c-means Clustering](https://guihongwan.github.io/2018/11/Fuzzy-C-means-Clustering/)
