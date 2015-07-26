'''
Training GMM model and SVM model for sensor data.
'''
__author__ = '1000892'
print(__doc__)

import os, sys, pickle
import numpy as np
from numpy import genfromtxt
from sklearn import svm, preprocessing

BASE_PATH = '/home/sibo/Documents/Projects/'
DATASET_PATH = BASE_PATH + 'yael/data_csv/'
LIBRARY_PATH = BASE_PATH + 'yael/yael_v438/'
RESULT_PATH = BASE_PATH + 'yael/results/'
TYPE = 'sensor'

sys.path.append(LIBRARY_PATH)
from yael import ynumpy

WINDOW_SIZE = 10

######################################################
# list of available images
sensor_names = [filename 
               for filename in os.listdir(DATASET_PATH)
               if filename.endswith('.csv')]

# sort data names
sensor_names.sort()

######################################################
# convert raw data into window-like data
all_data = []
for sname in sensor_names:
	# train with first 9 data in each activity
	# if not sname.endswith('10.csv'):
	print 'reading file:', sname
	# convert data into windows
	sensor_data = genfromtxt(DATASET_PATH + sname, delimiter=",")
	# delete first two columns
	# sensor_data = sensor_data[:,2:]
	nrow, ncol = sensor_data.shape
	nrow_window = nrow / WINDOW_SIZE
	ncol_window = ncol * WINDOW_SIZE
	# match row number with window_data
	sensor_data = sensor_data[0:(nrow_window * WINDOW_SIZE),:]
	window_data = []
	for col in range(ncol):
		window_data.append(sensor_data[:,col].reshape((-1, WINDOW_SIZE)))
	window_data = np.hstack(window_data)
	window_data = window_data.astype('float32')
	all_data.append(window_data)
	print window_data.shape
all_data_np = np.vstack(all_data)

######################################################
# GMM part
k = 25
n_sample = k * 100
# choose n_sample descriptors at random
sample_indices = np.random.choice(all_data_np.shape[0], n_sample)
sample = all_data_np[sample_indices]
# until now sample was in uint8. Convert to float32
sample = sample.astype('float32')
# compute mean and covariance matrix for the PCA
mean = sample.mean(axis = 0)
sample = sample - mean
cov = np.dot(sample.T, sample)
# compute PCA matrix and keep only 64 dimensions
eigvals, eigvecs = np.linalg.eig(cov)
# sort by increasing eigenvalue
perm = eigvals.argsort()
# eigenvectors for the half last eigenvalues
pca_transform = eigvecs[:, perm[-(WINDOW_SIZE*19)/2:]]
# transform sample with PCA (note that numpy imposes line-vectors,
# so we right-multiply the vectors)
sample = np.dot(sample, pca_transform)
# train GMM
gmm = ynumpy.gmm_learn(sample, k)

image_fvs = []
for image_desc in all_data:
   # apply the PCA to the image descriptor
   image_desc = np.dot(image_desc - mean, pca_transform)
   # compute the Fisher vector, using the derivative w.r.t mu and sigma
   fv = ynumpy.fisher(gmm, image_desc, include = 'mu, sigma')
   image_fvs.append(fv)

# make one matrix with all FVs
image_fvs = np.vstack(image_fvs)
# normalizations are done on all descriptors at once
# power-normalization
image_fvs = np.sign(image_fvs) * np.abs(image_fvs) ** 0.5
# L2 normalize
norms = np.sqrt(np.sum(image_fvs ** 2, 1))
image_fvs /= norms.reshape(-1, 1)

print image_fvs.shape

# save fisher vector to file
file_fv = open(RESULT_PATH + 'fv_' + TYPE + '.pkl', 'wb')
pickle.dump(image_fvs, file_fv)
file_fv.close()

######################################################
# SVM train part
C = 10
X_train = image_fvs
y_train = np.kron(np.arange(20), np.ones(10))

svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

######################################################
# save GMM model to file
file_gmm = open(RESULT_PATH + 'gmm_' + TYPE + '.pkl', 'wb')
pickle.dump(gmm, file_gmm)
file_gmm.close()

# save SVM model to file
file_svm = open(RESULT_PATH + 'svm_' + TYPE + '.pkl', 'wb')
pickle.dump(svc, file_svm)
file_svm.close()

# save mean and pca_transform to file
file_mean = open(RESULT_PATH + 'mean_' + TYPE + '.pkl', 'wb')
pickle.dump(mean, file_mean)
file_mean.close()

file_pca = open(RESULT_PATH + 'pca_' + TYPE + '.pkl', 'wb')
pickle.dump(pca_transform, file_pca)
file_pca.close()
