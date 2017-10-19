# -*- coding: utf-8 -*-
"""
TH 1 - Bài 4: Thực hiện Clustering cho dữ liệu Labeled Faces in the Wild
                    
                + Features: Facenet embedding
                        Paper: https://arxiv.org/pdf/1503.03832.pdf
                        REF Code: https://github.com/davidsandberg/facenet

                  Phiên bản dùng để visualize: 
                        + Chỉ lấy các samples có ít nhất 60 ảnh ( min_faces_per_person = 60 )
                        + Số lượng ảnh: 1348
                        + Số people = số clusters = 8

@author: Hoàng Hữu Tín - 14520956
Language: Python 3.6.1 - Anaconda 4.4 (64-bit)
OS: Windows 10 x64
Created on Wed Oct 18 08:30:25 2017
Last Modified: Oct 18 11:30 PM
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

# My custom file
from face_utils import face_embedding

#################################################
    # Define min number of facial images per person. (Only get data of people who have #images >= this number)
    # The smaller this number is, the larger dataset.
MIN_FACES_PER_PERSON = 60
##################################################

# Facenet Embeddings vector filename: Structure as a Dictionary with attributes
    # Embeddings: list of Facenet Embeddings vector for each image
    # Target: True label for each image
    # n_images: number of images current in this dataset
    # n_people: number of people = number of clusters
EMBED_filename = "facenet_embeddings_lfw_minface" + str(MIN_FACES_PER_PERSON) + ".data"


# Model files store model trained from clustering methods:  Structure as a Dictionary with attributes
    # model: trained model return from scikit learn functions
    # time: processing time
    # name: name of clustering method
KMEANS_filename = "Kmeans_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
SPECTRAL_filename = "Spectral_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
DBSCAN_filename = "DBSCAN_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"
AGGLO_filename = "Agglo_visual_minface" + str(MIN_FACES_PER_PERSON) + ".model"

# Structure of 2 dictionary will be stored in file
EMBED_DATA = {"Embeddings": [], "Target": [], "n_images":0, "n_people":0}         # for HIST data
CLUSTER = {"model":None, "time": 0, "name" : None}                              # for models file

# Directory path to lfw dataset need to get features
    # NOTE: THIS FOLDER MUST BE CONTAIN ONLY PEOPLE WITH MINFACE DEFINED
ALIGNED_LFW_PATHS = "./Dataset/lfw_mtcnnpy_160_minface_60"
# Path to facenet model
MODEL_DIR = "./facenet_model/20170512-110547"

# ======================================== Get Embeddings features =======================================
if (os.path.isfile(EMBED_filename)):
    # Use previous Embeddings Data
    with open(EMBED_filename, "rb") as fp:
        EMBED_DATA = pickle.load(fp)
    print("Load previous Embeddings data from file \"%s\"" % EMBED_filename)
    print("\t+ n_images = %d" % EMBED_DATA["n_images"])
    print("\t+ n_people = n_clusters = %d" % EMBED_DATA["n_people"])

else:
    # Create new Embedding Data
    print("Get Data from the labeled face in the Wild: run face_embedding.get_features_from_images_path()")

    # Use fetch function to get Target
    lfw_people = fetch_lfw_people(min_faces_per_person = MIN_FACES_PER_PERSON)
    N_IMAGES = len(lfw_people.images)
    EMBED_DATA["n_images"] = N_IMAGES
    EMBED_DATA["n_people"] = len(lfw_people.target_names)
    EMBED_DATA["Target"] = lfw_people.target
        
    EMBED_DATA["Embeddings"] = face_embedding.get_features_from_images_path(images_path = ALIGNED_LFW_PATHS, 
                                              model_dir = MODEL_DIR, file_type = 'png')

    print("\t+ n_images = %d" % EMBED_DATA["n_images"])
    print("\t+ n_people = n_clusters = %d" % EMBED_DATA["n_people"])

   
    print("\tSave Embeddings data to file \"%s\"" % EMBED_filename)
    # Save EMBED_DATA to file
    with open(EMBED_filename, "wb") as fp:
        pickle.dump(EMBED_DATA, fp)

# ============================================ Run Clustering Methods ========================================

N_CLUSTERS = EMBED_DATA["n_people"]
data = scale(EMBED_DATA["Embeddings"])        # normalize
TARGET = EMBED_DATA["Target"]

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
