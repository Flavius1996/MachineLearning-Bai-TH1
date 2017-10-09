# -*- coding: utf-8 -*-
"""
TH 1 - Bài 3: Thực hiện K-means cho dữ liệu Labeled Faces in the Wild dự trên feature LBP

                  Tranform data sang PCA 2-D trước khi chạy Clustering      

                  Phiên bản dùng để visualize: 
                        + Chỉ lấy các samples có ít nhất 70 ảnh ( min_faces_per_person = 70 )
                        + Số people = số cluster = 7


@author: Hoàng Hữu Tín - 14520956
Created on Sun Oct  8 16:24:10 2017
Last Modified: Oct 10 12:35 AM
"""
# import the necessary packages
from skimage import feature
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os.path

from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import scale

HIST_filename = "HIST_visual.data"
KMEANS_filename = "Kmeans_visual.model"

# Initialize Global Variable
RADIUS = 3
N_POINTS = RADIUS * 8

# List store all histogram of LBP images
HIST_DATA = {}
HIST_DATA["Histogram"] = []
HIST_DATA["Target"] = []
HIST_DATA["n_images"] = 0
HIST_DATA["n_people"] = 0

# ======================================== Get LBP hist Vectors =======================================
if (os.path.isfile(HIST_filename)):
    # Use previous Histogram Data
    with open(HIST_filename, "rb") as fp:
        HIST_DATA = pickle.load(fp)
    print("Load previous Histogram data from file \"%s\"" % HIST_filename)
    
else:
    # Create new Histogram Data
    print("Get Data from the labeled face in the Wild: run fetch_lfw_people()")
    # Get Data
    lfw_people = fetch_lfw_people(min_faces_per_person = 70)
    N_IMAGES = len(lfw_people.images)
    HIST_DATA["n_images"] = N_IMAGES
    HIST_DATA["n_people"] = len(lfw_people.target_names)
    
    #==============================================================================
    # # Visualize some image in Data 
    # fig = plt.figure(figsize=(8, 6))
    # # plot several images
    # for i in range(15):
    #     ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    #     ax.imshow(lfw_people.images[i], cmap=plt.cm.bone)
    #==============================================================================
    
    print("Computing Histogram of LBP images")
    # Loop over all images in Dataset
    for i in range(N_IMAGES):
        if (i % 3 == 0):
            print("\tProcessing image (%d/%d) ... %.2f %%"% (i, N_IMAGES, i*100/N_IMAGES))
    
        # Compute LBP image from original image
        lbp = feature.local_binary_pattern(lfw_people.images[i], N_POINTS,
    			RADIUS, method="uniform")
        
        # Create Histogram of LBP image, normalize by density (NOT probability mass)
        (hist, _) = np.histogram(lbp.ravel(), 	bins = int(lbp.max() + 1),
    			range=(0, N_POINTS + 2), density = True)
        
        # normalize the histogram - Use a probability mass function;
        # Comment bcz already use density = True above
    #		hist = hist.astype("float")
    #		hist /= (hist.sum() + 1e-7)
        HIST_DATA["Histogram"].append(hist)
        HIST_DATA["Target"].append(lfw_people.target[i])
    
    print("Done! 100%")
    print("Saved Hist data to file \"%s\"" % HIST_filename)
    # Save Hist_data to file
    with open(HIST_filename, "wb") as fp:
        pickle.dump(HIST_DATA, fp)
    
# ======================================== K-Means Clustering =======================================
print("Run K-MEANS clustering:")
N_CLUSTERS = HIST_DATA["n_people"]
data = scale(HIST_DATA["Histogram"])

reduced_data = PCA(n_components = 2).fit_transform(data)

if (os.path.isfile(KMEANS_filename)):
    # Use previous Training K-means model
    with open(KMEANS_filename, "rb") as fp:
        kmeans = pickle.load(fp)
    print("Load previous Kmeans model from file \"%s\"" % KMEANS_filename)
    
else:
    print("\tTraining K-MEANS model...")
    kmeans = cluster.KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=10).fit(reduced_data)
    
    with open(KMEANS_filename, "wb") as fp:
        pickle.dump(kmeans, fp)
    print("Saved Training Kmeans model to file \"%s\"" % KMEANS_filename)
     
y_pred_kmeans = kmeans.predict(reduced_data)

# ======================================== Spectral Clustering =======================================
#==============================================================================
# print("Run Spectral clustering")
# # Using Nearest Neighbors affinity
# connectivity = kneighbors_graph(data, n_neighbors=10, include_self=True, n_jobs=1)
# affinity_matrix = 0.5 * (connectivity + connectivity.T)
# 
# y_pred_spectral = cluster.SpectralClustering(affinity="precomputed",
#                                          n_clusters=N_CLUSTERS,
#                                          eigen_solver='arpack').fit_predict(affinity_matrix)
#==============================================================================

## ############################## SHOW RESULT ################################
print("Visualize Result")

# Step size of the mesh: step range from x/y_min -> x/y_max
h = .02

# Plot the decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
plt.title('K-means clustering on the LFW face data (PCA-reduced data)\n'
          'Centroids are marked with red cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()