'''
Training SVM model for video data.
'''
__author__ = '1000892'
print __doc__

import os, sys, pickle, time, datetime
import numpy as np
from sklearn import svm, cross_validation, preprocessing
from sklearn.metrics import confusion_matrix

BASE_PATH = '/home/sibo/Documents/Projects/'
DATASET_PATH = BASE_PATH + 'multimodal_sensors/video_data/feature/'
LIBRARY_PATH = BASE_PATH + 'yael/yael_v438/'
RESULT_PATH = BASE_PATH + 'multimodal_sensors/results/'
TYPE = 'video'

sys.path.append(LIBRARY_PATH)
from yael import ynumpy

#####################################################
# load fisher vector from file
file_fv = open(RESULT_PATH + 'fv_' + TYPE + '.pkl', 'rb')
image_fvs = pickle.load(file_fv)
file_fv.close()

#####################################################
# SVM train part
C = 100
X = image_fvs
y = np.concatenate((0*np.ones(10), 1*np.ones(10), 2*np.ones(10), 3*np.ones(10), 4*np.ones(10), 5*np.ones(10), 6*np.ones(10), 7*np.ones(10), 8*np.ones(10), 9*np.ones(10)))

C = 100.0  # SVM regularization parameter
TIMES = 10
NUM_OF_CLF = 10

cm_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_rbf_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
# cm_poly_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_lin_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
svc_score = 0
rbf_svc_score = 0
# poly_svc_score = 0
lin_svc_score = 0

# Preprocess the data
# X = preprocessing.scale(X)

for i in range(TIMES):

    print 'The ' + str(i) + ' iteration...'

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=np.random.RandomState())

    # print 'Training set shape:', X_train.shape, y_train.shape
    # print 'Test set shape:', X_test.shape, y_test.shape

    svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X_train, y_train)
    # poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, y_train)
    lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

    y_svc_pred = svc.predict(X_test)
    y_rbf_svc_pred = rbf_svc.predict(X_test)
    # y_poly_svc_pred = poly_svc.predict(X_test)
    y_lin_svc_pred = lin_svc.predict(X_test)

    cm_svc += confusion_matrix(y_test, y_svc_pred, range(NUM_OF_CLF))
    cm_rbf_svc += confusion_matrix(y_test, y_rbf_svc_pred, range(NUM_OF_CLF))
    # cm_poly_svc += confusion_matrix(y_test, y_poly_svc_pred, range(NUM_OF_CLF))
    cm_lin_svc += confusion_matrix(y_test, y_lin_svc_pred, range(NUM_OF_CLF))

    svc_score += svc.score(X_test, y_test)
    rbf_svc_score += rbf_svc.score(X_test, y_test)
    # poly_svc_score += poly_svc.score(X_test, y_test)
    lin_svc_score += lin_svc.score(X_test, y_test)

np.set_printoptions(precision=4, suppress=True)

print 'Linear SVC: ', svc_score / TIMES
print (cm_svc.T / cm_svc.sum(axis=1) ).T

print 'RBF SVC: ', rbf_svc_score / TIMES
print (cm_rbf_svc.T / cm_rbf_svc.sum(axis=1) ).T

# print 'Poly SVC: ', poly_svc_score / TIMES
# print (cm_poly_svc.T / cm_poly_svc.sum(axis=1) ).T

print 'LinearSVC: ', lin_svc_score / TIMES
print (cm_lin_svc.T / cm_lin_svc.sum(axis=1) ).T

# ######################################################
# # save SVM model to file
# file_svm = open(RESULT_PATH + 'svm_' + TYPE + '.pkl', 'wb')
# pickle.dump(svc, file_svm)
# file_svm.close()
