---  
tag: Clustering
---

Clustering algorithms can be classified as listed below:    
- Exclusive Clustering
- Overlapping Clustering
- Hierarchical Clustering    
- Mixture of Gaussians    

K-means is an exclusive algorithm. Fuzzy C-means is an overlapping clustering algorithm.  Hierarchical clustering is obvious and lasting Mixture of Gaussian is a probabilistic clustering algorithm.    


# Content

I'm going to talk about several algorithms, implement them or demo how to use them.
1. KMeans
   - [KMeans Clustering](https://guihongwan.github.io/2018/11/K-means-Clustering/)
   - [KMeans++ Clustering](https://guihongwan.github.io/2018/11/K-means++-Clustering/)
   - [KMeans Parallel](https://guihongwan.github.io/2018/11/K-means-Parallel/)
2. CMeans
   - [Fuzzy CMeans Clustering](https://guihongwan.github.io/2018/11/Fuzzy-C-means-Clustering/)
3. Hierarchical Clustering
4. Mixture of Gaussians
5. Spectral Clustering

# Distance Measure
Different formulas lead to different clusterings. Domain knowledge must be used to guide the formulation of a suitable distance measure for each particular application.

## Minkowski Metric
For higher dimensional data, a popular measure is the Minkowski metrix,
$$d_F(x_i, x_j) = (\sum_{k=1}^d \mid x_{i,k} - x_{j,k} \mid ^F)^{1 \over F}$$    
The Euclidean distance is a special case where $F=2$, while Manhattan metric has $F=1$.     
