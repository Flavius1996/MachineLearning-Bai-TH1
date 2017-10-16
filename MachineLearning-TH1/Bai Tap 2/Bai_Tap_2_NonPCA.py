# -*- coding: utf-8 -*-
"""
TH 1 - Bài 2: Thực hiện K-means, Spectral, DBSCAN và Agglomerative Clustering cho dữ liệu Hand-written Digits

            Không tranform data sang PCA 2-D trước khi chạy Clustering

@author: Hoàng Hữu Tín - 14520956
Language: Python 3.6.1 - Anaconda 4.4 (64-bit)
OS: Windows 10 x64
Created on Thu Sep 28 14:06:38 2017
Last Modified: Oct 10 21:20 PM
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import cluster, datasets, mixture
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import kneighbors_graph


# Global Variables
CLUSTER = {"model":None, "time": 0, "name" : None}
CLUSTERS_ARR = []

## ############################# READ DATA ###################################
np.random.seed(42)

digits = load_digits()
data = scale(digits.data)           # normalized data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
TARGET = digits.target

## ############################# K-MEANS ###################################
t0 = time()
kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(data)
CLUSTER["time"] = time() - t0
CLUSTER["name"] = "K-means"
CLUSTER["model"] = kmeans
CLUSTERS_ARR.append(CLUSTER.copy())

y_pred_kmeans = kmeans.predict(data)

## ############################# SPECTRAL ###################################
t0 = time()
connectivity = kneighbors_graph(data, n_neighbors=10, include_self=True, n_jobs=1)
affinity_matrix = 0.5 * (connectivity + connectivity.T)

spectral = cluster.SpectralClustering(affinity="precomputed",
                                         n_clusters=n_digits,
                                         eigen_solver='arpack').fit(affinity_matrix)
CLUSTER["time"] = time() - t0
CLUSTER["name"] = "Spectral"
CLUSTER["model"] = spectral
CLUSTERS_ARR.append(CLUSTER.copy())

y_pred_spectral = spectral.labels_

## ############################## DBSCAN ###################################
t0 = time()
db = cluster.DBSCAN(eps=0.3, min_samples=10).fit(data)

CLUSTER["time"] = time() - t0
CLUSTER["name"] = "DBSCAN"
CLUSTER["model"] = db
CLUSTERS_ARR.append(CLUSTER.copy())

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)      # Number of Clusters
y_pred_dbscan = db.labels_

## ########################### Agglomerative #################################
t0 = time()
agglo = cluster.AgglomerativeClustering(linkage="ward", n_clusters=n_digits).fit(data)
CLUSTER["time"] = time() - t0
CLUSTER["name"] = "Agglomerative"
CLUSTER["model"] = agglo
CLUSTERS_ARR.append(CLUSTER.copy())

y_pred_agglo = agglo.labels_

## ######################## PERFORMANCE EVALUATION  ############################
print(20 * ' '+ "NON PCA-BASED CLUSTERING PERFORMANCE EVALUATION")
print(84 * '=')
print('Method\t\ttime\tARI\tAMI\thomogeneity\tcompleteness\tv-measure')
print(84 * '-')
for method in CLUSTERS_ARR:
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%7.3f\t%16.3f\t%7.3f'
                % ( method["name"], method["time"],
                metrics.adjusted_rand_score(TARGET, method["model"].labels_),
                metrics.adjusted_mutual_info_score(TARGET,  method["model"].labels_),
                metrics.homogeneity_score(TARGET, method["model"].labels_),
                metrics.completeness_score(TARGET, method["model"].labels_),
                metrics.v_measure_score(TARGET, method["model"].labels_))
               # metrics.fowlkes_mallows_score(TARGET, method["model"].labels_))        # Seem don't work, return: nan
         )
print(84 * '=')
## ############################## SHOW RESULT ################################

# Reduce 64-dim data to 2-dim data for visualization
reduced_data = PCA(n_components=2).fit_transform(data)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
f.suptitle("Non PCA-based Clustering", fontsize=16)

# Plot the centroids as a red X
#centroids = PCA(n_components = 2).fit_transform(kmeans.cluster_centers_.data)
#ax1.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='r', zorder=10)

ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_kmeans, s = 4)
ax1.set_title('K-means')

ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_spectral, s = 4)
ax2.set_title('Spectral')

ax3.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_dbscan, s = 4)
ax3.set_title('DBSCAN')

ax4.scatter(reduced_data[:, 0], reduced_data[:, 1], c = y_pred_agglo, s = 4)
ax4.set_title('Agglomerative')

plt.show()

