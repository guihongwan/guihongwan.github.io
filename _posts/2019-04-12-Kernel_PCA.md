---  
tag: Data Representation 
---

# Introduction
- an example of kernel PCA implemented in sklearn.
- PCA
- Kernel PCA
- Implementation
- Derivation

# Kernel PCA Example


```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)
print(X.shape)
```

    (400, 2)



```python
# kernel PCA
# n_components : int, default=None. If None, all non-zero components are kept.
# kernel: default 'linear'
# gamma: kernel coefficient for rbf, poly and sigmoid kernels
# fit_inverse_transform: Learn the invser transform for non-precomputed kernels.
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
print('X_kpca.shape:', X_kpca.shape)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X)
print('X_pca.shape:', X_pca.shape)


# Plot results
plt.figure()
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(2, 2, 2, aspect='equal')
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 2, 3, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue",
            s=20, edgecolor='k')

plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplots_adjust(0.02, 0.10, 0.98, 1.3, 0.04, 0.35)

plt.show()
```

    X_kpca.shape: (400, 390)
    X_pca.shape: (400, 2)



<!-- ![png](output_3_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_1.png" width="300"/>



# Dimensionality Reduction

Linear: 
- $\bf{PCA}$: The aim is to find the subspace of largest variance, given the number of retained dimensions beforehand.        

Non-linear:    
- Kernel PCA    
- $\bf{Independent Component Analysis}$: there is interesting signal in the directions of small variance.      
- Multi-dimensional scaling    

# Principal Components Analysis

$\bf{PCA}$: unsupervised problem.       
$\bf{Linear  Discriminant Analysis}$: supervised     

PCA operates on zero-centered data:    
${1\over N}\sum_\limits{i} x_i = 0$    
covariance matrix C:    
$C={1 \over N }\sum_\limits{i} x_ix_i^T$    
It gives an eigen decomposition of C:    
$\lambda v = Cv$

The eigenvalues of C represent the variance in the eigen-directions of data-space.    
$C = V\Lambda V^T = \sum_\limits{a}\lambda_a v_a v_a^T$

The projection:    
$y_i = V_r^Tx_i \quad \forall i$    
where $V_r$ means the $d \times r$ sub-matrix containing the first $r$ eigenvectors as columns.

$\bf{Property \: 1}$: We can show that the projected data are de-correlated in the new basis:   
${1 \over N }\sum_\limits{i} y_i y_i^T = {1 \over N }\sum_\limits{i} V_r^Tx_i(V_r^Tx_i)^T = {1 \over N }\sum_\limits{i} V_r^Tx_i x_i^TV_r = V_r^T C V_r = V_r^T V\Lambda V^T V_r = \Lambda_r$    
where $\Lambda_r$ is the diagonal $r \times r$ submatrix corresponding to the largest eigenvalues.


$\bf{Property \: 2}$: the reconstruction error in $L_2$-norm from y to x is minimal:
$\sum_\limits{i} \parallel x_i - V_rV_r^Tx_i\parallel^2$

# Kernel PCA

To understand the utility of kernel PCA, particularly for clustering, observe that, while N points cannot in general be linearly separated in p dimensions, they may be linearly separated in higher dimensions.     
Given N points, $x_i$, the function $\phi$ maps the original p-dimensional features into a larger d-dipensional feature space.    
$x_i\rightarrow \phi(x_i)$    
But $\phi$ is never calculated explicitly($\phi$-space, called 'feature sapce'). Instead we work on the N-by-N kernel. Because we are never working directly in the feature space, the kernel-formulation of PCA is restricted in that it computes not the principal components themsevels, but the projectons of our data onto those components.     

## Steps for Kernel PCA
    
- Pick a kernel       
$K(x_i,x_j) = \phi(x_i)^T\phi(x_j)$    
- Center the kernel    
$\tilde{K} = K - 1_NK - K1_N + 1_NK1_N$

- Solve an eigenvalue problem    
$\tilde{K}a_i = \lambda_ia_i$    
$\lambda_i$ and $a_i$ are eigenvalues and eigenvectors of $\tilde{K}$. $\lambda_i = N\lambda$     
We need to make sure $\parallel a_i\parallel = {1 \over \sqrt{\lambda_i} }$

- For any data point, we can present it as    
$y_i = \sum_\limits{i=1}^na_{ji}K(x, x_i)$

## Problems
In linear PCA, we use the eigenvalues to rank the eigenvectors. It could be applied to KPCA. However, in practice there are cases that all variations of the data are same.     
If N is big, K is going to be large.

## Popular Kernels
Gaussian $K(x_i, x_j) = exp(-\gamma \parallel x_i - x_j\parallel^2)$    
$\gamma$ is a free parameter that is to be optitimized.

Polynomial $K(x_i, x_j) = (1+x_i^Tx_j)^p$

Hyperbokic tangent $K(x_i, x_j) = tanh(x_i^Tx_j+\delta)$

# Implementation


```python
# Data
Xt = X.T
plt.figure()
plt.subplot(1, 1, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
```




    <matplotlib.text.Text at 0x1a21d74048>




<!-- ![png](output_24_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_2.png" width="300"/>


## PCA


```python
import tool
B= Xt@Xt.T
D,V = tool.EVD(B)
# actually the assign does not matter, as long as they are consistent.
# here, only for the plots are same
V = -V
Xt_pca1 = V.T@Xt
X_pca1 = Xt_pca1.T

plt.subplot(1, 2, 1, aspect='equal')
plt.scatter(X_pca1[reds, 0], X_pca1[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_pca1[blues, 0], X_pca1[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(1, 2, 2, aspect='equal')
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA(lib)")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplots_adjust(0.0, 0.8, 0.98, 1.3, 0.04, 0.5)
```


<!-- ![png](output_26_0.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_3.png" width="300"/>


## Gaussian Radial Basis Function(RBF) Kernal PCA


```python
# kernel matrix
def kernel_rbf(Xt, gamma):
    '''
    Xt: a column is a point
    k(x_i, x_j) = exp(-gamma||x_i - x_j||^2)
    Xt is m by n.
    K will be n by n.
    '''
    import math
    m,n = Xt.shape
    K = np.zeros(n*n).reshape([n,n])
    for i in range(n):
        for j in range(n):
            x_i = Xt[:,i]
            x_j = Xt[:,j]
            p = -gamma * (np.linalg.norm(x_i-x_j))**2
            K[i,j] = math.e**p
    return K

def centralize_K(K):
    '''
    K may not be centered.
    Here, we center it.
    K_tilde = K - 1NK - K1N + 1NK1N
    1N is a N by N matrix with all values equal to 1/n.
    '''
    n,n = K.shape
    num = n*n
    IN = np.array([1/n]*num).reshape((n,n))
#     K_tilde = K - 2*IN@K + IN@K@IN
    K_tilde = K - IN@K - K@IN + IN@K@IN
    return K_tilde

def kPCA_eigen(K_tilde):
    import math
    D,a = tool.EVD(K_tilde)
    m,n = a.shape
    for i in range(n):
        if(D[i]) < 0:
            break
        scale = 1/(math.sqrt(D[i]))
        a[:,i] = scale*a[:,i]
        
    a = -a # this doesn't matter
    return (D,a)
    
def kPCA_projection(K_tilde, a):
    return(a.T@K_tilde)

def kernelPCA(Xt, kernel='rbf', gamma=10):
    K = kernel_rbf(Xt, gamma)
    K_tilde = centralize_K(K)
    D,a = kPCA_eigen(K_tilde)
    Xt_kpca = kPCA_projection(K_tilde, a)
    return Xt_kpca
    
X_kpca2 = kernelPCA(Xt).T
```


```python
plt.subplot(1, 2, 1, aspect='equal')
plt.scatter(X_kpca2[reds, 0], X_kpca2[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca2[blues, 0], X_kpca2[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component")
plt.ylabel("2nd component")

plt.subplot(1, 2, 2, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA(lib)")
plt.xlabel(r"1st principal component")
plt.ylabel("2nd component")

plt.subplots_adjust(0.0, 0.98, 1, 1.3, 0.04, 0.5)
```


<!-- ![png](output_29_0.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_4.png" width="300"/>


## More Application


```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=50, random_state=1)

# Data
Xt = X.T
plt.figure()
plt.subplot(1, 1, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1
plt.scatter(X[reds, 0], X[reds, 1], c='red', alpha=0.5)
plt.scatter(X[blues, 0], X[blues, 1], c="blue",alpha=0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
```




    <matplotlib.text.Text at 0x1a21d37eb8>




<!-- ![png](output_31_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_5.png" width="300"/>



```python
import tool
B= Xt@Xt.T
D,V = tool.EVD(B)
# actually the assign does not matter, as long as they are consistent.
# here, only for the plots are same
V = -V
Xt_pca1 = V.T@Xt
X_pca1 = Xt_pca1.T

plt.subplot(1, 1, 1, aspect='equal')
plt.scatter(X_pca1[reds, 0], X_pca1[reds, 1], c="red",alpha=0.5)
plt.scatter(X_pca1[blues, 0], X_pca1[blues, 1], c="blue",alpha=0.5)
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")
```




    <matplotlib.text.Text at 0x1a26f710b8>




<!-- ![png](output_32_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_6.png" width="300"/>



```python
X_kpca2 = kernelPCA(Xt, kernel='rbf', gamma=15).T
plt.subplot(1, 1, 1, aspect='equal')
plt.scatter(X_kpca2[reds, 0], X_kpca2[reds, 1], c="red",alpha=0.5)
plt.scatter(X_kpca2[blues, 0], X_kpca2[blues, 1], c="blue",alpha=0.5)
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component")
plt.ylabel("2nd component")
```




    <matplotlib.text.Text at 0x1a26f7ae48>




<!-- ![png](output_33_1.png) -->
<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/kernelpca_7.png" width="300"/>


# Derivation

Suppose that the mean of the data in the feature space is       
${1\over N}\sum_\limits{i} \phi(x_i) = 0$    
covariance matrix C:    
$C={1 \over N }\sum_\limits{i} \phi(x_i)\phi(x_i)^T$    
It gives an eigen decomposition of C:    
$\lambda v = Cv$    
But we do not directly work on $\lambda$ and $v$.

Eigenvectors can be expressed as linear combination of features:    
$$
v = \sum_\limits{i=1}^n a_{ri}\phi(x_i)
$$    
Proof:    
$$
Cv = {1 \over N }\sum_\limits{i} \phi(x_i)\phi(x_i)^Tv = \lambda v
$$    
thus    
$$
v = {1 \over \lambda N }\sum_\limits{i} \phi(x_i)\phi(x_i)^Tv = {1 \over \lambda N }\sum_\limits{i} \phi(x_i)^Tv\phi(x_i) = \sum_\limits{i} {\phi(x_i)^Tv \over \lambda N}\phi(x_i)
$$
We are able to show $xx^Tv = x^Tvx$, since $x^Tv$ is a scaler.     

Hence,    
$v = \sum_\limits{i} {a_{ri}}\phi(x_i)$, where $a_{ri} = {\phi(x_i)^Tv \over \lambda N}$.    
So, $\textbf{finding the eigenvectors is equivalent to finding the coefficients}$ $a_{r}$

Substituting $v = \sum_\limits{i} {a_{ri}}\phi(x_i)$ to ${1 \over N }\sum_\limits{i} \phi(x_i)\phi(x_i)^Tv = \lambda v$, we have:      
${1 \over N }\sum_\limits{i} \phi(x_i)\phi(x_i)^T\sum_\limits{l} {a_{rl}}\phi(x_l) = \lambda \sum_\limits{l} {a_{rl}}\phi(x_l)$    
${1 \over N }\sum_\limits{i} \phi(x_i)\sum_\limits{l} {a_{rl}}\phi(x_i)^T\phi(x_l) = \lambda \sum_\limits{l} {a_{rl}}\phi(x_l)$     
${1 \over N }\sum_\limits{i} \phi(x_k)^T\phi(x_i)\sum_\limits{l} {a_{rl}}\phi(x_i)^T\phi(x_l) = \lambda \sum_\limits{l} {a_{rl}}\phi(x_k)^T\phi(x_l)$    
${1 \over N }\sum_\limits{i} K(x_k,x_i)\sum_\limits{l} {a_{rl}}K(x_i,x_l) = \lambda \sum_\limits{l} {a_{rl}}K(x_k,x_l)$    
$K^2{a_r} = N\lambda {a_r}K$     
$K{a_r} = N\lambda {a_r}$     
$K{a_r} = \lambda_r {a_r}$, where $\lambda_r = N\lambda$.$a_r$ is a vector with dimensin $n \times 1$.     

Hence, $\textbf{finding $a_r$ is equivalent to finding the eigenvectors of $K$.}$

Since $v^Tv = 1$    
$\sum_\limits{i} ({a_{ri}}\phi(x_i))^T \sum_\limits{i} {a_{ri}}\phi(x_i) = \sum_\limits{i}\sum_\limits{j} {a_{ri}}{a_{rj}}\phi(x_i)^T \phi(x_j)$    
$a_r^TKa_r = 1$    
Since $K{a_r} = \lambda_r {a_r}\Rightarrow a_r^T\lambda_r {a_r} = 1 \Rightarrow \parallel a_r \parallel = {1\over \sqrt{\lambda_r}}$

At the begining, we suppose that    
${1\over N}\sum_\limits{i} \phi(x_i) = 0$    
But even with ${1\over N}\sum_\limits{i} x_i = 0$, we cannot guarantee that ${1\over N}\sum_\limits{i} \phi(x_i) = 0$    
We need to normalize the feature space.

$\tilde{\phi} (x_i) = \phi(x_i) - {1\over n}\sum_\limits{k=1}^n\phi(x_k)$       
$\tilde{K}(x_i,x_j) = \tilde{\phi}(x_i)^T\tilde{\phi}(x_j) = K(x_i,x_j) - {1\over n}\sum_\limits{k=1}^n K(x_i, x_k)- {1\over n}\sum_\limits{k=1}^n K(x_j, x_k) + {1\over n^2}\sum_\limits{l,k=1}^n K(x_l, x_k)$      
$\tilde{K} = K - 2I_{\frac{1}{n}}K + I_{1\over n}KI_{1\over n}$

 or     
 $\tilde{K}(x_i,x_j) = \tilde{\phi}(x_i)\tilde{\phi}(x_j)^T$    
$\tilde{K} = K -I_{1\over n}K - KI_{1\over n} + I_{1\over n}KI_{1\over n}$

# Reference  
1. https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html
2. http://rasbt.github.io/mlxtend/user_guide/feature_extraction/RBFKernelPCA/
3. https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
