'''
Training GMM model and SVM model for video data.
'''
__author__ = '1000892'
print __doc__

import os, sys, pickle, time, datetime
import numpy as np
from sklearn import svm

BASE_PATH = '/home/sibo/Documents/Projects/'
DATASET_PATH = BASE_PATH + 'multimodal_sensors/video_data/feature/'
LIBRARY_PATH = BASE_PATH + 'yael/yael_v438/'
RESULT_PATH = BASE_PATH + 'multimodal_sensors/results/'
TYPE = 'video'

sys.path.append(LIBRARY_PATH)
from yael import ynumpy

start_time = datetime.datetime.now()
print 'Start running:', start_time

######################################################
# function
def save_model(sample, type_feature, RESULT_PATH, TYPE):
	'''Save GMM models, mean, pca_transform of sample data

	Parameters
	----------
	sample: np.ndarray type,
		Sample data 
	type_feature: {'traj', 'hog', 'hof', 'mbh'}
		One of four types of feature
	RESULT_PATH: string
		Path of result files
	TYPE: {'video', 'sensor'}
		Type of data

	Returns
	-------
	gmm: gmm model
	mean: mean value of data
	pca_transform: pca transform of data
	'''
	# GMM part
	k = 25
	if type_feature == 'traj':
		DIM_AFTER_PCA = 0.5 * 30
	elif type_feature == 'hog':
		DIM_AFTER_PCA = 0.5 * 96
	elif type_feature == 'hof':
		DIM_AFTER_PCA = 0.5 * 108
	elif type_feature == 'mbh':
		DIM_AFTER_PCA = 0.5 * 192

	# compute mean and covariance matrix for the PCA
	mean = sample.mean(axis = 0)
	sample = sample - mean
	cov = np.dot(sample.T, sample)
	# compute PCA matrix and keep only 64 dimensions
	eigvals, eigvecs = np.linalg.eig(cov)
	# sort by increasing eigenvalue
	perm = eigvals.argsort()
	# eigenvectors for the 64 last eigenvalues
	pca_transform = eigvecs[:, perm[-DIM_AFTER_PCA:]]
	# transform sample with PCA (note that numpy imposes line-vectors,
	# so we right-multiply the vectors)
	sample = np.dot(sample, pca_transform)
	# train GMM
	gmm = ynumpy.gmm_learn(sample, k)
	# save mean value to file
	file_mean = open(RESULT_PATH + 'mean_' + type_feature + '_' + TYPE + '.pkl', 'wb')
	pickle.dump(mean, file_mean)
	file_mean.close()
	# save pca transform to file
	file_pca = open(RESULT_PATH + 'pca_' + type_feature + '_' + TYPE + '.pkl', 'wb')
	pickle.dump(pca_transform, file_pca)
	file_pca.close()
	# save GMM model to file
	file_gmm = open(RESULT_PATH + 'gmm_' + type_feature + '_' + TYPE + '.pkl', 'wb')
	pickle.dump(gmm, file_gmm)
	file_gmm.close()

	return gmm, mean, pca_transform

######################################################
# list of available images
video_names = [filename 
               for filename in os.listdir(DATASET_PATH)
               if filename.endswith('.feature')]

# sort file names
video_names.sort()

######################################################
# randomly select 1 percent data to form codebook
# (video feature numbers in each file is around 100,000) 
sample = []
for vname in video_names:
	# train with first 9 data in each activity
	if not vname.endswith('10.txt'):
		print 'reading file:', vname
		video_data = np.genfromtxt(DATASET_PATH + vname)
		# delete first ten columns
		video_data = video_data[:, 10:]
		nrow, ncol = video_data.shape
		print nrow, ncol

		sample_indices = np.random.choice(nrow, 0.01 * nrow)
		sample_each = video_data[sample_indices]
		sample_each = sample_each.astype('float32')
		
		sample.append(sample_each)

sample = np.vstack(sample)
print sample.shape

######################################################

sample_traj = sample[:, 0:30]
sample_hog  = sample[:, 30:126]
sample_hof  = sample[:, 126:234]
sample_mbh  = sample[:, 234:426]

# save gmm model, mean value, pca transform
gmm_traj, mean_traj, pca_traj = save_model(sample_traj, 'traj', RESULT_PATH, 'video')
gmm_hog, mean_hog, pca_hog = save_model(sample_hog, 'hog', RESULT_PATH, 'video')
gmm_hof, mean_hof, pca_hof = save_model(sample_hof, 'hof', RESULT_PATH, 'video')
gmm_mbh, mean_mbh, pca_mbh = save_model(sample_mbh, 'mbh', RESULT_PATH, 'video')

image_fvs = []
for vname in video_names:
	# train with first 9 data in each activity
	if not vname.endswith('10.txt'):
		print 'reading file:', vname
		video_data = np.genfromtxt(DATASET_PATH + vname)
		# delete first ten columns
		video_data = video_data[:,10:]
		video_data = video_data.astype('float32')
		# seperate data into different features
		video_data_traj = video_data[:, 0:30]
		video_data_hog  = video_data[:, 30:126]
		video_data_hof  = video_data[:, 126:234]
		video_data_mbh  = video_data[:, 234:426]
		# apply the PCA to the image descriptor
		video_data_traj = np.dot(video_data_traj - mean_traj, pca_traj)
		video_data_hog = np.dot(video_data_hog - mean_hog, pca_hog)
		video_data_hof = np.dot(video_data_hof - mean_hof, pca_hof)
		video_data_mbh = np.dot(video_data_mbh - mean_mbh, pca_mbh)
		# compute the Fisher vector, using the derivative w.r.t mu and sigma
		fv_traj = ynumpy.fisher(gmm_traj, video_data_traj, include = 'mu, sigma')
		fv_hog = ynumpy.fisher(gmm_hog, video_data_hog, include = 'mu, sigma')
		fv_hof = ynumpy.fisher(gmm_hof, video_data_hof, include = 'mu, sigma')
		fv_mbh = ynumpy.fisher(gmm_mbh, video_data_mbh, include = 'mu, sigma')
		# concatenate the fisher vectors
		fv = np.concatenate((fv_traj, fv_hog, fv_hof, fv_mbh))
		print fv.shape
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

#####################################################
# SVM train part
C = 100
X_train = image_fvs
# y_train = np.concatenate((0*np.ones(9), 1*np.ones(9), 2*np.ones(9), 3*np.ones(9), 4*np.ones(9), 5*np.ones(9), 6*np.ones(9), 7*np.ones(9), 8*np.ones(9), 9*np.ones(9)))
y_train = np.concatenate((0*np.ones(10), 1*np.ones(10), 2*np.ones(10), 3*np.ones(10), 4*np.ones(10), 5*np.ones(10), 6*np.ones(10), 7*np.ones(10), 8*np.ones(10), 9*np.ones(10)))

svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

######################################################
# save SVM model to file
file_svm = open(RESULT_PATH + 'svm_' + TYPE + '.pkl', 'wb')
pickle.dump(svc, file_svm)
file_svm.close()

#####################################################
# show running time 
end_time = datetime.datetime.now()
print 'Finish running:', end_time
print 'Total running time:', end_time - start_time
