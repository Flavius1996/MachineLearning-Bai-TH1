# -*- coding: utf-8 -*-
"""
TH 1 - Bài 4: Thực hiện Clustering cho dữ liệu Labeled Faces in the Wild

                  Thực hiện embedding cho face:
                                + Tìm landmarks (Sử dụng dlib) -> Rotate theo mắt sao cho 2 mắt nằm ngang
                                + Crop ảnh, chỉ lấy phần center của face.
                    

Language: Python 3.6.1 - Anaconda 4.4 (64-bit) + tensonflow + facenet
OS: Windows 10 x64
Created on Wed Oct 18 14:33:05 2017
Last Modified: Oct 18 10:05 PM

Special thanks to [REF]: kevinlu1211  (https://github.com/kevinlu1211/FacialClusteringPipeline)

"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from facenet.src import facenet
import matplotlib.image as mpimg
import glob
import os


import tensorflow as tf # NOTE: this has to use Tensorflow version 1.x

DEFAULT_MODEL_DIR = "./facenet_model/20170512-110547"
DEFAULT_DATASET_DIR = "./Dataset/lfw_mtcnnpy_160"

def load_embedding_layer_for_facenet(model_dir =  DEFAULT_MODEL_DIR):
    # Now utilize facenet to find the embeddings of the faces
    # Get the save files for the models
    meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            model_dir_exp = os.path.expanduser(model_dir)
            print("\tImporting graph ...")
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
            print("\tRestoring session ...")
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            find_embeddings = lambda img : sess.run(embeddings, feed_dict = {images_placeholder : img, phase_train_placeholder : False})
    return(find_embeddings)

def prewhiten(x):
    # just normalizing the image
    mean = np.mean(x) # mean of all elements
    std = np.std(x) # std of all elements
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size)) # get the max between the std, and 1/sqrt(number_of_all_elements)
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return(y)

def get_features_from_images_path(images_path = DEFAULT_DATASET_DIR, model_dir = DEFAULT_MODEL_DIR, file_type = 'png', batch_size = 100):
    '''
        Get facial embeddings, use netface trained model

        Requirement:
            + Images in images_path must be same alignment method with image trained in model (default = MTCNN).
            + Images in images_path must be same file size with image trained in model (default = 160x160).

        Default model "20170216-091149":
            + Dataset: aligned LFW using MTCNN
            + Image size: 160x160       
            For more info: https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1

        INPUT:
            images_path : path to images folder need to get embeddings.
            model_dir : facenet model
            file_type : same filetype as image in images_path
            batch_size : Number of images to process in a batch

        OUTPUT:
            (np.array): 128-dimensional space embeddings for each image in images_path. Shape: (n, 128) with n = #images

    '''
    
    # Get all image from images_path
    image_paths =  glob.glob(images_path + "/**/*." + file_type)
    n_images = len(image_paths)

    images = []
    print("Reading dataset...")
    for i in range(n_images):
        if (i % 10 == 0):
            print("\tReading image (%d/%d) ... %.2f %%"% (i, n_images, i*100/n_images), end = "\r")
        image = mpimg.imread(image_paths[i])

        prewhitened_image = prewhiten(image)
        images.append(prewhitened_image)

    # Load the embeddings layer from facenet
    print("Loading facenet model (this may be take a while) ...")
    embedding_layer = load_embedding_layer_for_facenet(model_dir)

    # Calculate the embeddings
    print("Calculate the embeddings...")
    n_images = len(images)
    n_batches = int(np.ceil(float(n_images)/batch_size))

    embeddings = np.zeros((n_images, 128))              # Store n embeddings vectors (128-d)

    print("\tNo. batch = {}".format(n_batches))
    print("\tBatch size = {} images".format(batch_size))

    # Find the embeddings
    for i in range(n_batches):
        print("\tProcessing batch {}/{}".format(i+1, n_batches), end = "\r")
        start = i * batch_size
        end = min((i + 1) * batch_size, n_images)

        # Get the embeddings
        embeddings[start:end, :] = embedding_layer(images[start:end])
        
    return embeddings 


# DEBUG: CREATE pickle file
#import pickle
#embed_arr = get_features_from_images_path()

#filename = 'embeddings.features'
#with open(filename, "wb") as fp:
#        pickle.dump(embed_arr, fp)