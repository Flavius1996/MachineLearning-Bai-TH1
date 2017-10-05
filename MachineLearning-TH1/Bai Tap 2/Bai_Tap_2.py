# -*- coding: utf-8 -*-
"""
Bài TH 2: Thực hiện K-means và Spectral Clustering cho dữ liệu Hand-written Digits

@author: Hoàng Hữu Tín - 14520956
Created on Thu Sep 28 14:06:38 2017
Last Modified: Oct 05 5:02 PM
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import kneighbors_graph


## ############################# READ DATA ###################################
digits = load_digits()
data = scale(digits.data)           # normalized data

n_samples, n_features = data.shape
#n_digits = len(np.unique(digits.target))
#labels = digits.target

## ############################# K-MEANS ###################################

kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(data)
y_pred_kmeans = kmeans.predict(data)

## ############################# SPECTRAL ###################################

connectivity = kneighbors_graph(data, n_neighbors=10, include_self=True, n_jobs=1)
affinity_matrix = 0.5 * (connectivity + connectivity.T)

y_pred_spectral = cluster.SpectralClustering(affinity="precomputed",
                                         n_clusters=n_digits,
                                         eigen_solver='arpack').fit_predict(affinity_matrix)

## ############################## SHOW RESULT ################################

# Reduce 64-dim data to 2-dim data for visualization
reduced_data = PCA(n_components=2).fit_transform(data)


## PLOT K-MEANS
plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_kmeans, s=2)
# Plot the centroids as a red X
centroids = PCA(n_components=2).fit_transform(kmeans.cluster_centers_.data)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)

plt.title('K-means clustering on the digits dataset\n'
          'Centroids are marked with red cross')

## PLOT SPECTRAL
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_spectral, s = 1)
plt.title('Spectral clustering on the digits dataset')

plt.show()

