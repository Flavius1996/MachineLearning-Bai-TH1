# -*- coding: utf-8 -*-
"""
TH 1 - Bài 3: Thực hiện Clustering cho dữ liệu Labeled Faces in the Wild dự trên feature LBP

                  Tranform data sang PCA 2-D trước khi chạy Clustering      

                  Phiên bản dùng để visualize: 
                        + Chỉ lấy các samples có ít nhất 60 ảnh ( min_faces_per_person = 60 )
                        + Số lượng ảnh: 1348
                        + Số people = số clusters = 8
                        + Kích thước ảnh: (62, 47)


@author: Hoàng Hữu Tín - 14520956
Language: Python 3.6.1 - Anaconda 4.4 (64-bit)
OS: Windows 10 x64
Created on Sun Oct  8 16:24:10 2017
Last Modified: Oct 10 11:05 PM
"""
# import the necessary packages
from skimage import feature
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os.path
from time import time

from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import scale

#################################################
    # Define min number of facial images per person. (Only get data of people who have #images >= this number)
    # The smaller this number is, the larger dataset.
MIN_FACES_PER_PERSON = 60
##################################################

# LBP Histogram vector filename: Structure as a Dictionary with attributes
    # Histogram: list of LBP histogram vector for each image
    # Target: True label for each image
    # n_images: number of images current in this dataset (with MIN_FACES_PER_PERSON was defined)
    # n_people: number of people = number of clusters
HIST_filename = "HIST_visual_minface"+ str(MIN_FACES_PER_PERSON) + ".data"


# Model files store model trained from clustering methods:  Structure as a Dictionary with attributes
    # model: trained model return from scikit learn functions
    # time: processing time
    # name: name of clustering method
KMEANS_filename = "Kmeans_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
SPECTRAL_filename = "Spectral_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
DBSCAN_filename = "DBSCAN_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
AGGLO_filename = "Agglo_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"

# Structure of 2 dictionary will be stored in file
HIST_DATA = {"Histogram": [], "Target": [], "n_images":0, "n_people":0}         # for HIST data
CLUSTER = {"model":None, "time": 0, "name" : None}                              # for models file

# LBP parameters
RADIUS = 3
N_POINTS = RADIUS * 8

# ======================================== Get LBP hist features =======================================
if (os.path.isfile(HIST_filename)):
    # Use previous Histogram Data
    with open(HIST_filename, "rb") as fp:
        HIST_DATA = pickle.load(fp)
    print("Load previous Histogram data from file \"%s\"" % HIST_filename)
    print("\t+ n_images = %d" % HIST_DATA["n_images"])
    print("\t+ n_people = n_clusters = %d" % HIST_DATA["n_people"])

else:
    # Create new Histogram Data
    print("Get Data from the labeled face in the Wild: run fetch_lfw_people()")
    # Get Data
    lfw_people = fetch_lfw_people(min_faces_per_person = MIN_FACES_PER_PERSON)
    N_IMAGES = len(lfw_people.images)
    HIST_DATA["n_images"] = N_IMAGES
    HIST_DATA["n_people"] = len(lfw_people.target_names)

    print("\t+ n_images = %d" % HIST_DATA["n_images"])
    print("\t+ n_people = n_clusters = %d" % HIST_DATA["n_people"])
    print("\t+ Image size =", lfw_people.images[0].shape)

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
        if (i % 10 == 0):
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
    
    print("\tDone (%d/%d) ... 100%%" % (N_IMAGES, N_IMAGES))
    print("\tSave Histogram data to file \"%s\"" % HIST_filename)
    # Save Hist_data to file
    with open(HIST_filename, "wb") as fp:
        pickle.dump(HIST_DATA, fp)
  

# ============================================ Run Clustering Methods ========================================

N_CLUSTERS = HIST_DATA["n_people"]
data = scale(HIST_DATA["Histogram"])        # normalize
TARGET = HIST_DATA["Target"]

# Use PCA to reduce data to 2D
reduced_data = PCA(n_components = 2).fit_transform(data)

METHODS_name = ["K-MEANS", "SPECTRAL", "DBSCAN", "AGGLOMERATIVE"]
METHODS_filename = [KMEANS_filename, SPECTRAL_filename, DBSCAN_filename, AGGLO_filename]

CLUSTERS_ARR = []

def Run_ClusterMethod(name, filename, data, nclusters):
    print("Run %s clustering:" % name)
    CLUSTER = {}
    if (name not in METHODS_name):
        raise ValueError('Undefined Method name! Method name must be: ' + ", ".join(METHODS_name))

    if (os.path.isfile(filename)):
        # Use previous model
        with open(filename, "rb") as fp:
            CLUSTER = pickle.load(fp)
        print("\tLoad previous %s model from file \"%s\"" % (name,filename))
    
    else:
        print("\tTraining %s model..." % name)
        t0 = time()
        if (name == METHODS_name[0]):
            model = cluster.KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=10).fit(data)
        elif (name == METHODS_name[1]):
            model = cluster.SpectralClustering(affinity="nearest_neighbors", n_clusters=nclusters,
                                                    eigen_solver='arpack').fit(data)
        elif (name == METHODS_name[2]):
            model = cluster.DBSCAN(eps=0.3, min_samples=10).fit(data)
        elif (name == METHODS_name[3]):
            model = cluster.AgglomerativeClustering(linkage="ward", n_clusters=nclusters).fit(data)
            
        CLUSTER["time"] = time() - t0
        CLUSTER["name"] = name
        CLUSTER["model"] = model

        with open(filename, "wb") as fp:
            pickle.dump(CLUSTER, fp)
        print("\tSaved %s model to file \"%s\"" % (name, filename))

    CLUSTERS_ARR.append(CLUSTER.copy())

    return CLUSTER["model"]

for i in range(len(METHODS_name)):
    Run_ClusterMethod(METHODS_name[i], METHODS_filename[i], reduced_data, N_CLUSTERS)

## ######################## PERFORMANCE EVALUATION  ############################
print("\n" + 20 * ' '+ "CLUSTERING PERFORMANCE EVALUATION")
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
print(84 * '=' + "\n")

## ############################## SHOW RESULT ################################
print("Visualize Result")

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
f.suptitle("Clustering on LFW data with min_face_per_person = " + str(MIN_FACES_PER_PERSON) + ", n_clusters = " + str(N_CLUSTERS) , fontsize=16)

Point_size = 10

ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], c = CLUSTERS_ARR[0]["model"].labels_, s = Point_size)
ax1.set_title('K-means')

ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c = CLUSTERS_ARR[1]["model"].labels_, s = Point_size)
ax2.set_title('Spectral')

ax3.scatter(reduced_data[:, 0], reduced_data[:, 1], c = CLUSTERS_ARR[2]["model"].labels_, s = Point_size)
ax3.set_title('DBSCAN')

ax4.scatter(reduced_data[:, 0], reduced_data[:, 1], c = CLUSTERS_ARR[3]["model"].labels_, s = Point_size)
ax4.set_title('Agglomerative')

plt.show()