---
layout: default
title: "Principal Component Analysis"
date:   2025-10-23
excerpt: "These are my notes on Principal Component Analysis (PCA), with the algorithm and implementation in Python."
---

# Principal Component Analysis

### Overview

PCA is an unsupervised learning algorithm that lets you take data with a lot of features (50, 1000, etc.) and reduce the number of features considerably, so you can plot and visualize it.

*Example 1:* Given many features of a car (like length, width, etc.), the PCA algorithm will recognize that the width doesn’t change all that much across instances, and will pick length

*Example 2*: Given two important features (height and length), we might take a combination of the two to capture the importance that the two features convey.

*PCA:* reduce many number of features down to two or three $z$ axes, essentially compressing the original features. 

### Algorithm

Before conducting the PCA algorithm steps, the data need to be normalized to have zero mean and scaled to account for extremely different scales (square ft. VS number of rooms). 

For a two-variable dataset, run an axis (z-axis, also the principal component) that the coordinates are projected upon. The principal component is the axis that captures the most amount of variation in the data.

Let’s say that for two variables, the PCA algorithm found the z-axis to be the vector 

$[0.71 \;\ 0.71]$. To find the projection of the point $(2,3)$ onto this principal component, we take the dot product of the two vectors:

$$
\begin{bmatrix}
2 \\ 3
\end{bmatrix}

\cdot
\begin{bmatrix}
0.71 \\ 0.71
\end{bmatrix} =
3.55
$$

This is finding the first principal component. We can also find the second, third, etc. components, which will be $90^\circ$to the previous axis.

### Implementation

1. Optional pre-processing — Perform feature scaling
2. “Fit” the data to obtain 2 or 3 new axes (principal components) —`fit` does mean normalization by default.
3. Optionally examine how much info/variance is explained by each principal component — given by `explain_variance_ratio`
4. Transform (project) the data onto the new axes — done by `transform`

```python
X = np.array([
    [1,1], [2,1], [3,2],
    [-1, -1], [-2, -1], [-3, -2]
])
pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio # 0.992 (kept 99.2% of the original variability)

# returns a 1-dim vector whose values are the projections and distances from the origin
X_trans_1 = pca_1.transform(X)
# reconstructs the original data  
X_reduced_1 = pca.inverse_transform(X_trans_1)
```