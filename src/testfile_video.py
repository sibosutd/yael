'''
Test video data using GMM and SVM models trained before.
Usage: 
'''
__author__ = '1000892'
print __doc__

import os, sys, pickle, time, datetime
import numpy as np
import sklearn as sk

BASE_PATH = '/home/sibo/Documents/Projects/'
DATASET_PATH = BASE_PATH + 'multimodal_sensors/video_data/feature_test/'
LIBRARY_PATH = BASE_PATH + 'yael/yael_v438/'
RESULT_PATH = BASE_PATH + 'multimodal_sensors/results/'
TYPE = 'video'

sys.path.append(LIBRARY_PATH)
from yael import ynumpy

start_time = datetime.datetime.now()
print 'Start running:', start_time

######################################################
# load model
file_gmm = open(RESULT_PATH + 'gmm_' + TYPE + '.pkl', 'rb')
gmm = pickle.load(file_gmm)

file_svm = open(RESULT_PATH + 'svm_' + TYPE + '.pkl', 'rb')
svc = pickle.load(file_svm)

file_mean = open(RESULT_PATH + 'mean_' + TYPE + '.pkl', 'rb')
mean = pickle.load(file_mean)

file_pca = open(RESULT_PATH + 'pca_' + TYPE + '.pkl', 'rb')
pca_transform = pickle.load(file_pca)

######################################################
# test part
filename = sys.argv[-1]
print 'test file:\n\t', filename

sensor_data = genfromtxt(filename)
# delete first two columns
sensor_data = sensor_data[:,2:]
nrow, ncol = sensor_data.shape
nrow_window = nrow / WINDOW_SIZE
ncol_window = ncol * WINDOW_SIZE
# match row number with window_data
sensor_data = sensor_data[0:(nrow_window * WINDOW_SIZE),:]
window_data = []
for col in range(ncol):
	window_data.append(sensor_data[:,col].reshape((-1, 10)))
window_data = np.hstack(window_data)
window_data = window_data.astype('float32')

# generate fisher vector using GMM model
image_desc = np.dot(window_data - mean, pca_transform)
fv = ynumpy.fisher(gmm, image_desc, include = 'mu, sigma')
fvs = [fv]
# normalizations are done on all descriptors at once
# power-normalization
fvs = np.sign(fvs) * np.abs(fvs) ** 0.5
# L2 normalize
norms = np.sqrt(np.sum(fvs ** 2, 1))
fvs /= norms.reshape(-1, 1)

X_test = fvs[0]
y_svc_pred = svc.predict(X_test)

print 'classification result: \n\tactivity no.:', int(y_svc_pred[0]) + 1

#####################################################
# show running time 
end_time = datetime.datetime.now()
print 'Finish running:', end_time
print 'Total running time:', end_time - start_time