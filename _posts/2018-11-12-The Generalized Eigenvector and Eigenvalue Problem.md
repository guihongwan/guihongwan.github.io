---  
tag: Data Representation 
---

# Problem

Given two matrices A, B, our goal is to compute generalized eigenvalues $\lambda$ and the coresponding generalized eigenvectors v that satisfy the following fundamental equation:
$Av = \lambda Bv$    
This reduces to the standard eigenvalue problem with the choice B = I.    
Here, we only discuss cases where A is symmetric positive semidefinite and B is symmetric positive definite.


# Algorithm
$Av = \lambda Bv$     
$B^{-1}Av = \lambda v$     
We cannot simply get $\lambda v$ by eigh($B^{-1}A$), since $B^{-1}A$ is not symmetric.In practice, we may use eig($B^{-1}A$) to get the eigenvalues eigenvectors.     
    

Let $B = QDQ^T$ be the EVD of B, $v = QD^{-1/2}u$     
This gives:     
$ AQD^{-1/2}u    
= \lambda QDQ^TQD^{-1/2}u    
= \lambda QDD^{-1/2}u       
= \lambda QD^{1/2}u$     
$ D^{-1/2}Q^TAQD^{-1/2}u = \lambda u$
$ D^{-1/2}Q^TAQD^{-1/2}$ is positive semidefinte and symmetric, hence we can solve the corresponding regular eigenvalue problem for u and then convert back to v.     

Input: matrices A, B      
Output: generalized eigenvectors v1, v2, ..., vn, generalized eigenvalues $\lambda 1,\lambda 2, ..., \lambda n$

1. Compute the EVD: $B = QDQ^T$    
2. Compute the matrix $ C =  D^{-1/2}Q^TAQD^{-1/2}$    
3. Compute the EVD of C: $C = U∑U^T$    
4. Eigenvalues: ∑
5. Eigenvector: $v_j = QD^{-1/2}u_j$ 

# Implementation


```python
def generalized_Eigenh(A, B, r):
    '''
    Av = lambda Bv
    A should be positive semidefinite
    B should be positive definite
    '''
    import numpy as np
    D,Q = np.linalg.eigh(B)

    D = D**(-1/2)
    D_hat = np.diag(D)
    C = D_hat@Q.T@A@Q@D_hat

    S,U = np.linalg.eigh(C)
    
    sorted_idxes = np.argsort(-S)
    S = S[sorted_idxes]
    U = U[:, sorted_idxes]

    S = S[0:r]
    U = U[:,0:r]

    V = Q@D_hat@U
    
    return (S, V)
```
