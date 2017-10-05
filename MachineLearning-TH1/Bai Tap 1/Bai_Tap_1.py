# -*- coding: utf-8 -*-
"""
Bài TH 1 - Bài 1: Thực hiện K-means Clustering cho 2 Gaussian Blobs

@author: Hoàng Hữu Tín - 14520956
Created on Thu Sep 28 14:04:35 2017
Last Modified: Oct 05 4:25 PM
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(None)

plt.figure(figsize=(20, 20))

# Random 2 Gaussian Blob
centers = 2
n_samples = 2000            # 2000 points
random_state = np.random.randint(0, 200)
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)

# Run K-means (with k=2)
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

# Show Result
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Run K-means on 2 Gaussian Blobs")

plt.show()

