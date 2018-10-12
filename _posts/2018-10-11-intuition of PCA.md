
# TASK
- Find linear combinations that maximize variance subject to being uncorrelated with those already selected.    
- Find k-dimensional projection where $1\leq k \leq d$

$X_{dxn}=[x_1, x_2,...,x_n]$    
$x_i$ is dx1 column vector    

Assume X is mean-centered     

# Projection Definition

Let v be a dx1 column vector    
$v$ = $\begin{bmatrix} 
v_{1}\\
v_{2}\\
... \\
v_{d}
\end{bmatrix}
$    

$x$ = $\begin{bmatrix} 
x_{1}\\
x_{2}\\
... \\
x_{d}
\end{bmatrix}
$    

## projection
Projection of x onto v is the linear combination:    
$v^Tx$=$\begin{bmatrix} 
v_{1}, v_{2},...,v_{d}
\end{bmatrix}
$
$\begin{bmatrix} 
x_{1}\\
x_{2}\\
... \\
x_{d}
\end{bmatrix}
$ 
= $\sum\limits_{i=1}^d v_ix_i$      

$X_{dxn}=[x_1, x_2,...,x_n]$    
    
Projection of X onto v is $(X^Tv)^T = v^TX$:    
- an 1xn row vector
- a set of scalar values corresponding to n projected points  

## variance along projection
Variance along v is:    
$\delta_v^2$     
= $(v^TX)(v^TX)^T$    
= $v^TXX^Tv$    
= $v^TBv$, where $B = XX^T$ is hte dxd covariance matrix of the data since X has zero mean.

## Maximization of Variance    
Maximizing varance along v is not well-defined since we can increase it without limit by increasing the size of the components of v.    
Impose a normalization constraint on the v such that $v^Tv = 1$


Optimization problem is to maximize u = $v^TBv - \lambda(v^Tv-1)$    
where $\lambda$ is a Lagrange multiplier.   



Differentiating wrt v yelds:    
$\partial u \over \partial v$ = $2Bv - 2\lambda v$ = 0    
which reduces to     
$(B - \lambda I)v$ = 0     or     
Bv = $\lambda v$    

Hence v is eigenvectors, $\lambda$ is associated eigenvalues.    

## Approximation of X on V

$X_{dxn}=[x_1, x_2,...,x_n]$      
    
Projection of X onto v is $(X^Tv)^T = v^TX$:    
- an 1xn row vector
- a set of scalar values corresponding to n projected points 

$X_{dxn}=[x_1, x_2,...,x_n]$     
$V_{dxk}=[v_1, v_2,...,v_k]$     
Project of X onto V is $(X^TV)^T = V^TX$       
- an kxn matrix
- n k-dimentional points

