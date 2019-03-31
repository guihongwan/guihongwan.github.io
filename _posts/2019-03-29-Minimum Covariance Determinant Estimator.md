---  
tag: Data Representation 
---

# Introduction
The minimum covariance determinant(MCD) method is a highly robust estimator of multivariate location and scatter. Its objective is to find h observations(out of n) whose covariance matrix has the lowest determinant.     
Being resistant to outlying observations makes the MCD very useful for outlier detection.     
The statistical distance:    
$$d(x,\mu,S) = \sqrt{(x-\mu)'S^{-1}(x-\mu)}$$
$\mu$ is a vector, and $S$ is a positive definite $p\times p$.    

A multivariate distribution is called elliptically symmetric and unimodal if there exists a strictly decreasing real function $g$ such that the density can be written in the form:
$$f(x) = {1\over \sqrt{\mid S \mid}}g(d^2(x, \mu, S)$$

## Distance     
$\textbf{Mahalanobis distance}$: should tell us how far away $x$ is from the center of the data cloud    
$MD(x,\bar{x},Cov(X)) = \sqrt{(x-\bar{x})'Cov(X)^{-1}(x-\bar{x})}$    
$\mu$ is the arithmetic mean, and $Cov(X)$ is the covariance matrix.    


Mahalanobis distances are known to suffer from masking(strongly affected by contamination).     

$\textbf{A robust distance}$:     
$$RD(x,\hat{x}_{MCD},\hat{S}_{MCD})= d(x,\hat{x}_{MCD},\hat{S}_{MCD})$$   

$\hat{x}_{MCD}$     
is the MCD estimate of location and 
$S_{MCD}$    
is the MCD covariance estimate.

# Definition
The raw MCD estimator with tuning constant $n/2 \leq h \leq n$ is ($\hat{\mu}_0, \hat{S}_0$)
- the location estimate $\hat{\mu}_0$ is the mean of the h observations for which the determinant of the sample covariance matrix is as small as possible;
- the scatter matrix estimate $\hat{S}_0$ is the corresponding covariance matrix multiplied by a consistency factor $c_0$.     

Note: the MCD estimator can only be computed when h>p, so we need n>2p. To avaoid excessive noise, it is, however, recommended that n > 5p, so that we have at least five obserations per dimension. Otherwise, we can use Minimum Regularized Covariance Determinant method.    
The MCD estimator is the most robust when taking h = [(n + p + 1)/2], at the population level $\alpha = .5$.    

But the MCD suffers from low efficiency. For example, if $\alpha$=0.5 the asymptotic relative efficiency of the diagonal elements of the MCD scatter matrix relative to the sample covariance matrix is only 6% when p=2.    
In order to increase the efficiency while retaining high robustness one can apply a weighting step. For the MCD, this yields the estimates:    

$$\hat{\mu}_{MCD} = {\sum_{i=1}^n W(d_i^2)x_i \over \sum_{i=1}^n W(d_i^2)}$$

$$\hat{S}_{MCD} = c_1{1\over n}\sum_{i=1}^nW(d_i^2)(x_i-\hat{S}_{MCD})(x_i-\hat{S}_{MCD})'$$

A simple effective choisce for $W$ is to set it to 1 when the robust distance below the cutoff  $\sqrt{X_{p,0.975}^2}$


```R
sqrt(qchisq(.975, df=13))
```


4.97349021160508


# Experiment


```R
wine.fl <- "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(wine.fl,header = F)
colnames(wine) <- c('Type', 'Alcohol', 'Malic', 'Ash', 
                      'Alcalinity', 'Magnesium', 'Phenols', 
                      'Flavanoids', 'Nonflavanoids',
                      'Proanthocyanins', 'Color', 'Hue', 
                      'Dilution', 'Proline')
dim(wine)
```
178 14

```R
wine.nolabel = wine[,2:14]
dim(wine.nolabel)
head(wine.nolabel)
```


178 13
<table>
<tbody>
	<tr><td>14.23</td><td>1.71 </td><td>2.43 </td><td>15.6 </td><td>127  </td><td>2.80 </td><td>3.06 </td><td>0.28 </td><td>2.29 </td><td>5.64 </td><td>1.04 </td><td>3.92 </td><td>1065 </td></tr>
	<tr><td>13.20</td><td>1.78 </td><td>2.14 </td><td>11.2 </td><td>100  </td><td>2.65 </td><td>2.76 </td><td>0.26 </td><td>1.28 </td><td>4.38 </td><td>1.05 </td><td>3.40 </td><td>1050 </td></tr>
	<tr><td>13.16</td><td>2.36 </td><td>2.67 </td><td>18.6 </td><td>101  </td><td>2.80 </td><td>3.24 </td><td>0.30 </td><td>2.81 </td><td>5.68 </td><td>1.03 </td><td>3.17 </td><td>1185 </td></tr>
	<tr><td>14.37</td><td>1.95 </td><td>2.50 </td><td>16.8 </td><td>113  </td><td>3.85 </td><td>3.49 </td><td>0.24 </td><td>2.18 </td><td>7.80 </td><td>0.86 </td><td>3.45 </td><td>1480 </td></tr>
	<tr><td>13.24</td><td>2.59 </td><td>2.87 </td><td>21.0 </td><td>118  </td><td>2.80 </td><td>2.69 </td><td>0.39 </td><td>1.82 </td><td>4.32 </td><td>1.04 </td><td>2.93 </td><td> 735 </td></tr>
	<tr><td>14.20</td><td>1.76 </td><td>2.45 </td><td>15.2 </td><td>112  </td><td>3.27 </td><td>3.39 </td><td>0.34 </td><td>1.97 </td><td>6.75 </td><td>1.05 </td><td>2.85 </td><td>1450 </td></tr>
</tbody>
</table>




```R
wine.first = subset(wine, Type == '1')
wine.subset.first = data.frame(Malic = wine.first$Malic, Proline=wine.first$Proline)
dim(wine.subset.first)

# write.csv(wine.subset.first, file = "Wine1_nolabel.csv")
```

59 2



```R
# MCD
# install.packages('robustbase')
library(robustbase)
mcd = covMcd(wine.subset.first)
plot(mcd, which = "tolEllipsePlot", classic = TRUE)
```


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/mcd_0.png" width="300"/>



```R
par(mfrow = c(1,2))
plot(mcd, which = "distance", classic = TRUE)
```


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/mcd_1.png" width="300"/>



```R
plot(mcd, which = "dd", classic = TRUE)
```


<img src="https://github.com/guihongwan/guihongwan.github.io/raw/master/_posts/mcd_2.png" width="300"/>



```R
# conda install -c r r-pcapp
# conda install -c bioconda r-rrcov
```


```R
# library(rrcov)
# mcd = CovMrcd(wine.nolabel)
```


```R

```
