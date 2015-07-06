'''
Multi-classification
Training on pure data sets, test on pure data sets.
'''
__author__ = '1000892'

print __doc__

import numpy as np
from numpy import genfromtxt
from sklearn import svm, cross_validation, preprocessing
from sklearn.metrics import confusion_matrix

X_sensor = genfromtxt('/home/sibo/Documents/Projects/multimodal_sensors/results/fv_sensor.csv', delimiter=',') 
X_video = genfromtxt('/home/sibo/Documents/Projects/multimodal_sensors/results/fv_video.csv', delimiter=',')
# X = np.concatenate((X_sensor, X_video), axis=1)
# X = X_sensor
X = X_video
X = X[:,0:750] # Trajectory 2*25*15=750
# X = X[:,750:3150] # HOG 2*25*48=2400
# X = X[:,3150:5850] # HOF 2*25*54=2700
# X = X[:,5850:10650] # MBH 2*25*96=4800
print 'Fisher Data shape:', X.shape

y = np.concatenate((0*np.ones(10), 1*np.ones(10), 2*np.ones(10), 3*np.ones(10), 4*np.ones(10), 5*np.ones(10), 6*np.ones(10), 7*np.ones(10), 8*np.ones(10), 9*np.ones(10)))
print 'Groudtruth shape:', y.shape

C = 100.0  # SVM regularization parameter
TIMES = 100
NUM_OF_CLF = 10

cm_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_rbf_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_poly_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_lin_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
svc_score = 0
rbf_svc_score = 0
poly_svc_score = 0
lin_svc_score = 0

# X = preprocessing.scale(X)
# # print X.mean(axis=0)
# # print X.std(axis=0)
# X = X[0:800,:]

for i in range(TIMES):

    print 'The ' + str(i) + ' iteration...'

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=np.random.RandomState())

    # print 'Training set shape:', X_train.shape, y_train.shape
    # print 'Test set shape:', X_test.shape, y_test.shape

    svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X_train, y_train)
    # poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, y_train)
    lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

    y_svc_pred = svc.fit(X_train, y_train).predict(X_test)
    # y_rbf_svc_pred = rbf_svc.fit(X_train, y_train).predict(X_test)
    # y_poly_svc_pred = poly_svc.fit(X_train, y_train).predict(X_test)
    y_lin_svc_pred = lin_svc.fit(X_train, y_train).predict(X_test)

    cm_svc += confusion_matrix(y_test, y_svc_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # cm_rbf_svc += confusion_matrix(y_test, y_rbf_svc_pred, [0, 1, 2])
    # cm_poly_svc += confusion_matrix(y_test, y_poly_svc_pred, [0, 1, 2])
    cm_lin_svc += confusion_matrix(y_test, y_lin_svc_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    svc_score += svc.score(X_test, y_test)
    # rbf_svc_score += rbf_svc.score(X_test, y_test)
    # poly_svc_score += poly_svc.score(X_test, y_test)
    lin_svc_score += lin_svc.score(X_test, y_test)

np.set_printoptions(precision=4, suppress=True)

print 'Linear SVC: ', svc_score / TIMES
# print cm_svc
print (cm_svc.T / cm_svc.sum(axis=1) ).T
# print 'RBF SVC: ', rbf_svc_score / TIMES
# # print cm_rbf_svc
# print (cm_rbf_svc.T / cm_rbf_svc.sum(axis=1) ).T
# print 'Poly SVC: ', poly_svc_score / TIMES
# # print cm_poly_svc
# print (cm_poly_svc.T / cm_poly_svc.sum(axis=1) ).T
print 'LinearSVC: ', lin_svc_score / TIMES
# print cm_lin_svc
print (cm_lin_svc.T / cm_lin_svc.sum(axis=1) ).T
