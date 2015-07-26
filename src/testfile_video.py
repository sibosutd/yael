'''
Test video data using GMM and SVM models trained before.
'''
__author__ = '1000892'
print __doc__

from config import *

TYPE = 'video'

######################################################
# show processing time
# start_time = datetime.datetime.now()
# print 'Start running:', start_time

######################################################
# load model
# for type_feature in ['traj', 'hog', 'hof', 'mbh']:
file_gmm = open(MODEL_PATH + 'gmm_' + 'traj' + '_' + TYPE + '.pkl', 'rb')
gmm_traj = pickle.load(file_gmm)
file_gmm.close()

file_mean = open(MODEL_PATH + 'mean_' + 'traj' + '_' + TYPE + '.pkl', 'rb')
mean_traj = pickle.load(file_mean)
file_mean.close()

file_pca = open(MODEL_PATH + 'pca_' + 'traj' + '_' + TYPE + '.pkl', 'rb')
pca_traj = pickle.load(file_pca)
file_pca.close()

file_gmm = open(MODEL_PATH + 'gmm_' + 'hog' + '_' + TYPE + '.pkl', 'rb')
gmm_hog = pickle.load(file_gmm)
file_gmm.close()

file_mean = open(MODEL_PATH + 'mean_' + 'hog' + '_' + TYPE + '.pkl', 'rb')
mean_hog = pickle.load(file_mean)
file_mean.close()

file_pca = open(MODEL_PATH + 'pca_' + 'hog' + '_' + TYPE + '.pkl', 'rb')
pca_hog = pickle.load(file_pca)
file_pca.close()

file_gmm = open(MODEL_PATH + 'gmm_' + 'hof' + '_' + TYPE + '.pkl', 'rb')
gmm_hof = pickle.load(file_gmm)
file_gmm.close()

file_mean = open(MODEL_PATH + 'mean_' + 'hof' + '_' + TYPE + '.pkl', 'rb')
mean_hof = pickle.load(file_mean)
file_mean.close()

file_pca = open(MODEL_PATH + 'pca_' + 'hof' + '_' + TYPE + '.pkl', 'rb')
pca_hof = pickle.load(file_pca)
file_pca.close()

file_gmm = open(MODEL_PATH + 'gmm_' + 'mbh' + '_' + TYPE + '.pkl', 'rb')
gmm_mbh = pickle.load(file_gmm)
file_gmm.close()

file_mean = open(MODEL_PATH + 'mean_' + 'mbh' + '_' + TYPE + '.pkl', 'rb')
mean_mbh = pickle.load(file_mean)
file_mean.close()

file_pca = open(MODEL_PATH + 'pca_' + 'mbh' + '_' + TYPE + '.pkl', 'rb')
pca_mbh = pickle.load(file_pca)
file_pca.close()

# save svm model
file_svm = open(MODEL_PATH + 'svm_' + TYPE + '.pkl', 'rb')
svc = pickle.load(file_svm)
file_svm.close()

######################################################
# test part
filename = sys.argv[-1]
print 'test file:\n\t', filename

video_data = np.genfromtxt(filename)
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

# #####################################################
# # show running time 
# end_time = datetime.datetime.now()
# print 'Finish running:', end_time
# print 'Total running time:', end_time - start_time