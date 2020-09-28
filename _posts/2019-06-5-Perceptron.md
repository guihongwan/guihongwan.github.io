---  
tag: Neural Nets 
---

Perceptron is a one layer neural-nets.           
Output $F(x) = g(h)$, where $h = \sum_{i=1}^N w_ix_i$, g is the activation function and X is $N\times n$.    
n features means there are n nodes of perceptron.

Loss function:
$$L(w) = {1\over N} \sum_{i=1}^N (y_i - F(x_i))^2$$

We can use stochastic gradient descent(SGD) as a mean to train neural networks. SGD can be performed using 'mini-batches' of size B. When B=1, it is online gradient descent.    

In experiment, we are going to    
- compare the performance of different B.
- compare the performance of different activation functions.

# Implementation


```python
%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import math
import numpy as np
```


```python
def sigmoid(z):
    return 1/(1+np.exp(-2*z)) 

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative_relu(z):
    my_dict = {True:1, False:0}
    z_bool = z>0
    deri = np.vectorize(my_dict.get)(z_bool)
    return deri
```


```python
class Perceptron:
    DEBUG = False
    def __init__(self, activation=None, name=""):
        self.name = name
        self.activation = activation
        
    def initialize(self, n_weights):
        '''
        n_weights: number of nodes
        '''
        self.weight = np.ones(n_weights)
        self.weight = self.weight.dot(0.5) #inital guess, eg. [w0,w1,w2] = [0.5 0.5 0.5]
        if(self.DEBUG):
            print('inital weight:', self.weight)
        self.err = [] #store the err for each step of SGD
        
    def reset_err(self):
        self.err = []
        
    def feed(self, X, y):
        self.X = X
        self.y = y
        
    def out(self):
        h = self.X.dot(self.weight.T)

        if(self.activation):
            self.output = self.activation(h)
        else:
            self.output = h
        if(self.DEBUG):
            print('self.output:', self.output)
        return self.output
    
    def caculate_derivative(self):
        self.out() # update output
        if(self.activation is tanh):
            self.derivative = 1 - self.output**2
        elif(self.activation is relu):
            self.derivative = derivative_relu(self.output)
        elif(self.activation is sigmoid):
            self.derivative = self.output*(1-self.output)
        else:#linear unit
            self.derivative = 1
        if(self.DEBUG):
            print('self.derivative:\n',self.derivative)
        
    def updateWeights(self, e=0.1):
        e_list = self.y[:,0] - self.output
        e_list = e_list * self.derivative
        B,n = self.X.shape
        NGB = self.X.T@e_list
        NGB = NGB/B
        
        self.weight = self.weight + e*NGB
        
        if(self.DEBUG):
            print('new weight:\n', self.weight, '\nshape:\n', self.weight.shape)
            
        # Loss
        self.out()
        e_list = self.y[:,0] - self.output
        self.err.append(sum([x**2 for x in e_list])/len(e_list))
        
    def predict(self, instance):
        h = sum(instance*self.weight)
        
        if(self.activation):#not None
            return self.activation(h)
        else:
            return h
        
    def getError(self):
        return self.err
    
    def getWeight(self):
        return self.weight
```

# Compare B


```python
# generate Data
N = 100
n = 10

X = np.random.normal(0, 1, n*N).reshape((N,-1))
a = np.random.normal(0, 1, n).reshape((n,-1))
noise = np.random.normal(0, 0.01, N).reshape((N,-1))
y = X@a + noise
```


```python
def TrainPerceptron(X, B, epochs, activation=None):
    # initialize
    perceptron = Perceptron(activation)
    n_weights = len(X[0,])#the number of weight equals to the number of features
    perceptron.initialize(n_weights)

    # Training
    N,n = X.shape
    for epoch in range(epochs):
        for t in range(0,N,B):
            XB = X[t:(t+B),]
            yB = y[t:(t+B),]
            perceptron.feed(X, y)
            perceptron.out()
            perceptron.caculate_derivative()
            perceptron.updateWeights(e=0.1)       

    # After Training
    w = perceptron.getWeight()
    err= perceptron.getError()
    return(w, err)
```


```python
epochs = 10
B = 1
w, err = TrainPerceptron(X, B, epochs, activation=None)
print('final weight:',w)
print('True weight:', a.T)

plt.figure(figsize=(10,5))
plt.xlabel("Step of SGD")
plt.ylabel("Loss")
z = np.linspace(0, len(err), len(err))
plt.scatter(z, err, linewidth=2)

# error at end of each epoch
err1 = []
T = N//B
for i in range(T-1, len(err)+1, T):
    err1.append(err[i])
```

    final weight: [ 1.80781894 -1.58183054 -1.14671566  0.75034867 -1.00353975  0.29815968
     -0.34168361  0.52653896  0.61069221 -0.58223247]
    True weight: [[ 1.80706745 -1.58129345 -1.14600803  0.74867477 -1.00394042  0.29702684
      -0.34012387  0.52698682  0.60870624 -0.58302009]]



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/perceptron_1.png" width="300"/>




```python
epochs = 10
B = 5
w, err = TrainPerceptron(X, B, epochs, activation=None)
print('final weight:',w)
print('True weight:', a.T)

plt.figure(figsize=(10,5))
plt.xlabel("Step of SGD")
plt.ylabel("Loss")
z = np.linspace(0, len(err), len(err))
plt.scatter(z, err, linewidth=2)

# error at end of each epoch
err5 = []
T = N//B
for i in range(T-1, len(err)+1, T):
    err5.append(err[i])
```

    final weight: [ 1.80782311 -1.58183155 -1.14671193  0.75035292 -1.00353569  0.2981578
     -0.34167913  0.52654143  0.61069768 -0.58223064]
    True weight: [[ 1.80706745 -1.58129345 -1.14600803  0.74867477 -1.00394042  0.29702684
      -0.34012387  0.52698682  0.60870624 -0.58302009]]



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/perceptron_2.png" width="300"/>



```python
epochs = 10
B = 10
w, err = TrainPerceptron(X, B, epochs, activation=None)
print('final weight:',w)
print('True weight:', a.T)

plt.figure(figsize=(10,5))
plt.xlabel("Step of SGD")
plt.ylabel("Loss")
z = np.linspace(0, len(err), len(err))
plt.scatter(z, err, linewidth=2)

# error at end of each epoch
err10 = []
T = N//B
for i in range(T-1, len(err)+1, T):
    err10.append(err[i])
```

    final weight: [ 1.80872058 -1.5818729  -1.14601618  0.75131259 -1.00229425  0.29744306
     -0.34018977  0.52722288  0.61164628 -0.58157357]
    True weight: [[ 1.80706745 -1.58129345 -1.14600803  0.74867477 -1.00394042  0.29702684
      -0.34012387  0.52698682  0.60870624 -0.58302009]]



<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/perceptron_3.png" width="300"/>




```python
z = np.linspace(0, epochs, epochs)
plt.plot(z, err1, linewidth=2, marker='x', label='B=1')
plt.plot(z, err5, linewidth=2, marker='x', label='B=5')
plt.plot(z, err10, linewidth=2, marker='x', label='B=10')
plt.legend()
```




    <matplotlib.legend.Legend at 0x11ed2f278>




<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/perceptron_4.png" width="300"/>


# Compare activation function


```python
epochs = 10
B = 1
w, err_linear = TrainPerceptron(X, B, epochs, activation=None)
w, err_sigmoid = TrainPerceptron(X, B, epochs, activation=sigmoid)
w, err_tanh = TrainPerceptron(X, B, epochs, activation=tanh)
w, err_relu = TrainPerceptron(X, B, epochs, activation=relu)
```


```python
plt.figure(figsize=(10,5))
plt.xlabel("Step of SGD")
plt.ylabel("Loss")
z = np.linspace(0, len(err_linear), len(err_linear))

end = len(err_linear) - 900
plt.plot(z[:end], err_linear[:end], label='linear')
plt.plot(z[:end], err_sigmoid[:end], label='sigmoid')
plt.plot(z[:end], err_tanh[:end], label='tanh')
plt.plot(z[:end], err_relu[:end], label='relu')
plt.legend()
```




    <matplotlib.legend.Legend at 0x11df37630>




<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/perceptron_5.png" width="300"/>




```python

```
