---  
tag: Computer Vision 
---

# Introduction

Template matching is a technique in digital image processing for finding small parts of an image which match a template image.    
It can be used as a way to detect edeges in images.     
## Feature-based approach
Feature-based approach relies on the extraction of image features such,i.e. shapes, textures, colors, to match the target or frame. This approach is currently achieved by using Neural Networks and Deep learning classifiers such as VGG, ResNet.    
## Template-based approach    
For templates without strong features, or for when the bulk of the template image constitutes the matching image, a template-based approach may be effective.   
Here, we are going to introduce basic methods of template matching using cross correlation.     

In order to detect a template $\lambda$ in a picture $f$, we can compute the cross correlation of $\lambda$ with $f$. Large values correspond to a probable match.    
    
Let $w$ be a window in the picture of the same dimension as $\lambda$.    
In vector notation, the cross-correlation match measure at the location of $w$ is $\lambda \cdot w$.   
A good match measure should be invariant to scaling. We can achieve this by computing the cosine of angle between $\lambda$ and $w$. It is given by $\lambda \cdot w \over \mid \lambda \mid \mid w\mid$. We can compute $\lambda \cdot w \over \mid w\mid$, since for each window, $\mid \lambda \mid$ is same.

The above introductions results in two algorithms, non-normalized template matching and normalized template matching.    
### Non-normalized template matching
for each pixel(i,j) in $f$ compute the matching measure $Q(i,j)$ by 
$$
Q(i,j) = \sum_{\alpha, \beta in \lambda} \lambda(\alpha, \beta) f(i+\alpha, j+\beta)
$$

### Normalized template matching
for each pixel(i,j) in $f$ compute the matching measure $Q(i,j)$ by 
$$
Q(i,j) = {\sum_{\alpha, \beta in \lambda} \lambda(\alpha, \beta) f(i+\alpha, j+\beta) \over
\sqrt{\sum_{\alpha, \beta in \lambda} \lambda(\alpha, \beta) f^2(i+\alpha, j+\beta)}}
$$


# Implementation


```python
import numpy as np
from scipy import signal
```


```python
def nonNormalizedMatching(temple, pic):
    Q = signal.correlate2d(temple, pic)
    return Q
def normalizedMatching(temple, pic):
    corr = signal.correlate2d(temple, pic)

    pic_square = pic*pic
    nrow_t, ncolumn_t = temple.shape
    templeate1 = np.ones([nrow_t, ncolumn_t])
    N = signal.correlate2d(templeate1, pic_square)

    N =np.sqrt(N)
    Q = corr/N
    return Q
```

## test


```python
nrow = 3
ncolumn = 3

temple = np.zeros([nrow, ncolumn])
for j in range(ncolumn):
    temple[1,j] = 1
print('temple:\n',temple)

print()
pic1 = np.ones([nrow, ncolumn])
print('image1:\n', pic1)
print()
pic2 = np.ones([1, ncolumn])
print('image2:\n', pic2)
```

    temple:
     [[0. 0. 0.]
     [1. 1. 1.]
     [0. 0. 0.]]
    
    image1:
     [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    
    image2:
     [[1. 1. 1.]]



```python
print()
no_normRet1 = nonNormalizedMatching(temple, pic1)
print(no_normRet1)
print()
no_normRet2 = nonNormalizedMatching(temple, pic2)
print(no_normRet2)
```

    
    [[0. 0. 0. 0. 0.]
     [1. 2. 3. 2. 1.]
     [1. 2. 3. 2. 1.]
     [1. 2. 3. 2. 1.]
     [0. 0. 0. 0. 0.]]
    
    [[0. 0. 0. 0. 0.]
     [1. 2. 3. 2. 1.]
     [0. 0. 0. 0. 0.]]



```python
print()
normRet1 = normalizedMatching(temple, pic1)
print(normRet1)
print()
normRet2 = normalizedMatching(temple, pic2)
print(normRet2)
```

    
    [[0.         0.         0.         0.         0.        ]
     [0.70710678 1.         1.22474487 1.         0.70710678]
     [0.57735027 0.81649658 1.         0.81649658 0.57735027]
     [0.70710678 1.         1.22474487 1.         0.70710678]
     [0.         0.         0.         0.         0.        ]]
    
    [[0.         0.         0.         0.         0.        ]
     [1.         1.41421356 1.73205081 1.41421356 1.        ]
     [0.         0.         0.         0.         0.        ]]



```python

```
