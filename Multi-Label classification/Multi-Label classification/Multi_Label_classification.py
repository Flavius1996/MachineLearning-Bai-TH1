# -*- coding: utf-8 -*-
"""
Bài TH 2: Multi Label classification
                Using   DecisionTreeClassifier
                        RandomForestClassifier
@author: Hoàng Hữu Tín - 14520956
Language: Python 3.6.1 - Anaconda 4.4 (64-bit)
OS: Windows 10 x64
Created on Thu Dec 15 08:15:36 2017
REF:
    http://scikit-learn.org/stable/auto_examples/plot_multilabel.html#sphx-glr-auto-examples-plot-multilabel-py
    http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.multiclass import OneVsRestClassifier

COLORS = np.array(['!',
                   '#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])

RANDOM_SEED = np.random.randint(2 ** 10)


CLASSIFIERS = [
    ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
    ]


# Generate multilabel dataset

def Create_MultiLabel_Dataset(n_samples = 100, n_labels=1, n_classes=3, n_features = 2, length=50, allow_unlabeled = False, bVisualize = False):

    # Use sklearn make_multilabel_classification 
    X, Y, p_c, p_w_c = make_multilabel_classification(n_samples=n_samples, n_features=n_features,
                                   n_classes=n_classes, n_labels=n_labels,
                                   length=length, allow_unlabeled=allow_unlabeled,
                                   return_distributions=True,
                                   random_state=RANDOM_SEED)

    if (bVisualize == True):
        plt.figure(figsize=(8, 6))

        plt.scatter(X[:, 0], X[:, 1], color=COLORS.take((Y
                                                        ).sum(axis=1)),
                   marker='.')
        plt.scatter(p_w_c[0] * length, p_w_c[1] * length,
                   marker='*', linewidth=.5, edgecolor='black',
                   s=20 + 1500 * p_c ** 2,
                   color=COLORS.take([1, 2, 4]))
        plt.xlabel('Feature 0 count')
        plt.ylabel('Feature 1 count')

        plt.title('n_labels={0}, n_classes={1}, n_samples={2}, length={3}'.format(n_labels, n_classes, n_samples, length ))
        plt.show()

        print('The data was generated from (random_state=%d):' % RANDOM_SEED)
        print('Class', 'P(C)', 'P(w0|C)', 'P(w1|C)', sep='\t')
        for k, p, p_w in zip(['red', 'blue', 'yellow'], p_c, p_w_c.T):
            print('%s\t%0.2f\t%0.2f\t%0.2f' % (k, p, p_w[0], p_w[1]))


    return X, Y, p_c, p_w_c

def Run_Classification(X, Y, bVisualize = True):
    
    figure = plt.figure(figsize=(27, 9))
    i = 1
 
    # PCA transform
    X = PCA(n_components=2).fit_transform(X)
    
    # Split Train - Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.4, random_state=42)


    h = .02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, len(CLASSIFIERS) + 1, i)
    ax.set_title("Test groundtruth")
    # Plot the training points
    zero_class = np.where(y_test[:, 0])
    one_class = np.where(y_test[:, 1])
    plt.scatter(X_test[:, 0], X_test[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X_test[zero_class, 0], X_test[zero_class, 1], s=160, edgecolors='b',
                    facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X_test[one_class, 0], X_test[one_class, 1], s=80, edgecolors='orange',
                    facecolors='none', linewidths=2, label='Class 2')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    i += 1
    for (name, clf) in CLASSIFIERS:
        ax = plt.subplot(1, len(CLASSIFIERS) + 1, i)
        
        classif = OneVsRestClassifier(clf)
        classif.fit(X_train, y_train)

        y_pred = classif.predict(X_test)

        plt.subplot(1, len(CLASSIFIERS) + 1, i)
        plt.title(name + 'predict')
    
        zero_class = np.where(y_pred[:, 0])
        one_class = np.where(y_pred[:, 1])
        plt.scatter(X_test[:, 0], X_test[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
        plt.scatter(X_test[zero_class, 0], X_test[zero_class, 1], s=160, edgecolors='b',
                    facecolors='none', linewidths=2, label='Class 1')
        plt.scatter(X_test[one_class, 0], X_test[one_class, 1], s=80, edgecolors='orange',
                    facecolors='none', linewidths=2, label='Class 2')
        
        
        plt.xticks(())
        plt.yticks(())
        
        i += 1

    if bVisualize == True:
        plt.tight_layout()
        plt.show()

#==============================================================================
# def plot_hyperplane(clf, min_x, max_x, linestyle, label):
#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
#     yy = a * xx - (clf.intercept_[0]) / w[1]
#     plt.plot(xx, yy, linestyle, label=label)
#     
# def Run_Classification_2(X, Y, bVisualize = True):
#     
#     # PCA transform
#     X = PCA(n_components=2).fit_transform(X)
# 
#     min_x = np.min(X[:, 0])
#     max_x = np.max(X[:, 0])
# 
#     min_y = np.min(X[:, 1])
#     max_y = np.max(X[:, 1])
# 
#     i_subplot = 2
#     for (name, clf) in CLASSIFIERS:
#         classif = OneVsRestClassifier(clf)
#         classif.fit(X, Y)
#     
#         plt.subplot(1, len(CLASSIFIERS) + 1, i_subplot)
#         plt.title(name)
#     
#         zero_class = np.where(Y[:, 0])
#         one_class = np.where(Y[:, 1])
#         plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
#         plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
#                     facecolors='none', linewidths=2, label='Class 1')
#         plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
#                     facecolors='none', linewidths=2, label='Class 2')
#     
#         #plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
#         #                'Boundary\nfor class 1')
#         #plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
#         #                'Boundary\nfor class 2')
#         plt.xticks(())
#         plt.yticks(())
#     
#         plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
#         plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
#         if i_subplot == 2:
#             plt.xlabel('First principal component')
#             plt.ylabel('Second principal component')
#             plt.legend(loc="upper left") 
#         
#         i_subplot += 1
# 
#     if bVisualize == True:
#         plt.show()
#==============================================================================

X, Y, p_c, p_w_c = Create_MultiLabel_Dataset(n_samples = 100, n_labels=1, n_classes=2, n_features=2, bVisualize = True)

Run_Classification(X, Y, bVisualize = True)
