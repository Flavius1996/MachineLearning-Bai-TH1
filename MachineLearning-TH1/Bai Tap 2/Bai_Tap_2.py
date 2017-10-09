# -*- coding: utf-8 -*-
"""
TH 1 - Bài 2: Thực hiện K-means, Spectral Clustering, DBSCAN cho dữ liệu Hand-written Digits

            Tranform data sang PCA 2-D trước khi chạy Clustering

@author: Hoàng Hữu Tín - 14520956
Created on Thu Sep 28 14:06:38 2017
Last Modified: Oct 10 12:30 AM
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
n_digits = len(np.unique(digits.target))
target = digits.target

# Tranform to PCA 2-D space
reduced_data = PCA(n_components=2).fit_transform(data)

## ############################# K-MEANS ###################################

kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(reduced_data)
y_pred_kmeans = kmeans.predict(reduced_data)

## ############################# SPECTRAL ###################################

connectivity = kneighbors_graph(reduced_data, n_neighbors=10, include_self=True, n_jobs=1)
affinity_matrix = 0.5 * (connectivity + connectivity.T)

y_pred_spectral = cluster.SpectralClustering(affinity="precomputed",
                                         n_clusters=n_digits,
                                         eigen_solver='arpack').fit_predict(affinity_matrix)

## ############################## DBSCAN ###################################
db = cluster.DBSCAN(eps=0.3, min_samples=10).fit(reduced_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
y_pred_dbscan = db.labels_
db_n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

## ############################## SHOW RESULT ################################

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
f.suptitle("PCA-based Clustering", fontsize=16)

# Plot the centroids as a red X
centroids = kmeans.cluster_centers_
ax1.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)

ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_kmeans, s = 4)
ax1.set_title('K-means')

ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_spectral, s = 4)
ax2.set_title('Spectral')

ax3.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_dbscan, s = 4)
ax3.set_title('DBSCAN')

plt.show()

