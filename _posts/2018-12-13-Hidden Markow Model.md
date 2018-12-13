---  
tag: NLP 
---

# Introduction
## Components:
- $\pi $ initial probability distribution which specifies the probabilities from $q_0$ to each state.
- Q = $q_1,q_2,...,q_n$
- a start state $q_0$ and a final state $q_f$
- O = $o_1,o_2,...,o_T$
- A: transition matrix. $a_{ij}$ means the probability from $q_i$ to $q_j$
- B = ${P(x_i=o_t \given q_i)}$, the probabilities of all possible $o_t$ when in the state of some $q_i$

## HMM For Three Basic Problems
### Likelihood
Given $\lambda = (A, B)$, and an observation sequence O, determine the likelihood $p(O|\lambda)$
### Decoding
Given $\lambda = (A, B)$, and an observation sequence O, discover the best hidden state sequence Q.
### Learning
Given O and the set of states, learn the HMM parameters A and B.    
The following part will cover the first two problems.

# Forward Algorithm for Likelihood
Given $\lambda = (A, B)$, O    
Determine the likelihood $p(O|\lambda)$

## Intuition
step1:    
P(O, Q'): joint probability of being in a state sequence Q' and in the Q', a particular sequence of observations O.    
$P(O, Q') = P(Q')P(O|Q')  $  
= $\prod_{t=1}^T P(q_t|q_{t-1})P(o_t|q_t)$      
    
step2:    
There are many possible Q' for O.        
hence, P$(O|\lambda)$ = $\sum_Q' P(O,Q')$

## Forword Algorithm: Dynamic Programming
$ \alpha_t(j) $: the probability of being in state j after seeing the first t observations.
$ \alpha_t(j) $    
$ = P(o_1,o_2,...,o_t, q_j|\lambda$    
$ = \sum_{i=1}^N \alpha_{t-1}(i)a_{ij}b_j(o_t)$



- Initialization:    
$ \alpha_1(j) = a_{0j}b_j(o_1)$ 1<= j <= N

- Recursion    
$ \alpha_t(j) = \sum_{i=1}^N \alpha_{t-1}(i)a_{ij}b_j(o_t)$

- Terminaton    
$p(O|\lambda) = \sum_{i=1}^N \alpha_T(i)a_{i,q_f}$

# Viterbi Algorithm for Decoding
Given $\lambda = (A, B)$, O    
Find the most possible sequence of states $Q'= q_1, q_2,...,q_T$

$V_t(j)$:the probability that HMM is in state j after seeing the first t observations.     
$V_{t-1}(j)$:the previous Viterbi path probability.    
$V_t(j) = \max_{i=1}^N V_{t-1}(i)a_(ij)b_j(o_t)$

## Implementation


```python
def viterbi(obs, A, B):
    '''
    Input:
    -obs: observations
    -A: transition probability matrix, including q0 and qf
    -B: Probability of an observation given state.
    The column index indicates the state, for example B[i, j] means 
    given qi, the possibility of j corresponding observation in vacabulary.
    Output:
    A path with probability
    '''
    DEBUG = False
    R = A.shape[0] # A already includes the q0 qf.
    N = R-2
    T = len(obs)
    
    # Initialization
    import numpy as np
    V = np.zeros(R*T).reshape((R, T))
    BP = np.zeros(R*T).reshape((R, T))
    o0 = obs[0]
    for s in range(1, N+1): # 1,2,...,N
        V[s, 0] = A[0,s]*B[o0, s]
        BP[s, 0] = 0
    
    # Recursion
    for t in range(1, T):
        for s in range(1, N+1):
            o = obs[t]
            all = [V[ss, t-1]*A[ss,s]*B[o, s] for ss in range(1, N+1)]
            V[s, t] = max(all)
            BP[s, t] = 1 + np.argmax(all) # +1, since argmax start from 0
        
    # Termination
    # in our case the A[s,f] = 1
    final_list = [V[s,T-1]*A[s,N+1] for s in range(1, N+1)]
    probability = max(final_list)
    final_s = 1+np.argmax(final_list)
    
    V[N+1, T-1] = probability
    BP[N+1, T-1] = final_s
    
    if DEBUG:
        print(V)
        print(BP)
    path = constructPath(BP)
    
    return(path, probability)
    
def constructPath(BP):
    R,T = BP.shape
    N = R-2
    final = BP[N+1, T-1]
    final = int(final)
    path = str(final)
    
    t = T-1
    prevous = final
    while(t > 0):
        prevous = BP[prevous, t]
        prevous = int(prevous)
        path = str(prevous) + path
        t = t-1
    return path
```

## TestCase


```python
import numpy as np

# transition probability Matrix 
A = np.array([
  #  q0  q1   q2  qf
    [0, 0.8, 0.2, 0], #q0
    [0, 0.7, 0.3, 1], #q1  1 here means it is able to stop at q1
    [0, 0.4, 0.6, 1], #q2  1 here means it is able to stop at q2
    [0, 0,   0,   0]
])
B = np.array([
  #  q0  q1   q2
    [0, 0,   0  ],   # 0
    [0, 0.2, 0.5], # 1
    [0, 0.4, 0.4], # 2
    [0, 0.4, 0.1]  # 3 
])
```


```python
obs = [3,3,1,1,2,2,3,1,3]

path, probability = viterbi(obs, A, B)
print(path, 'with probability', probability)
# Correct result
# V_m1 = np.array([
#   #  3        3          1          1           2            2           3             1             3
#     [0,       0,         0,         0,          0,           0,          0,            0,            0],  #q0
#     [0.32(0), 0.0896(1), 0.0125(1), 0.00176(1), 0.000645(2), 0.00018(1), 5.058e-05(1), 7.081e-06(1), 1.983e-06(1)],  #q1
#     [0.02(0), 0.0096(1), 0.0134(1), 0.00403(2), 0.000968(2), 0.00023(2), 1.393e-05(2), 7.587e-06,    4.5512e-07]  #q2
#  ]) 
# print('The sequences should be: 112211111 with likelihood 1.983e-06')
```

    112211111 with probability 1.9826343936e-06

<br>

Reference：Speech and Language Processing An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, SECOND EDITION.

<br>
For reproduction, please specify：[GHWAN's website](https://guihongwan.github.io) » [Hidden Markow Model](https://guihongwan.github.io/2018/12/Hidden-Markow-Model/)
