'''
Multi-classification
'''
__author__ = '1000892'
print(__doc__)

import pickle, csv
import numpy as np
from sklearn import svm, cross_validation, preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

BASE_PATH = '/home/sibo/Documents/Projects/'
RESULT_PATH = BASE_PATH + 'yael/results/'
TYPE = 'all'

if TYPE == 'all':
    file_video_fv = open(RESULT_PATH + 'fv_' + 'video' + '.pkl', 'rb')
    X_1 = pickle.load(file_video_fv)
    file_sensor_fv = open(RESULT_PATH + 'fv_' + 'sensor' + '.pkl', 'rb')
    X_2 = pickle.load(file_sensor_fv)
    X = np.concatenate((X_1, X_2), axis=1)
else:
    file_fv = open(RESULT_PATH + 'fv_' + TYPE + '.pkl', 'rb')
    X = pickle.load(file_fv)

print 'Fisher Data shape:', X.shape

C = 10.0  # SVM regularization parameter
TIMES = 100
NUM_OF_CLF = 20

y = np.kron(np.arange(NUM_OF_CLF), np.ones(10))
print 'Groudtruth shape:', y.shape

cm_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
cm_lin_svc = np.zeros((NUM_OF_CLF, NUM_OF_CLF))
svc_score = np.zeros(TIMES)
lin_svc_score = np.zeros(TIMES)

# X = preprocessing.scale(X)
# print X.mean(axis=0)
# print X.std(axis=0)

for i in range(TIMES):

    print 'The ' + str(i) + ' iteration...'

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=np.random.RandomState())

    # print 'Training set shape:', X_train.shape, y_train.shape
    # print 'Test set shape:', X_test.shape, y_test.shape

    svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

    y_svc_pred = svc.predict(X_test)
    y_lin_svc_pred = lin_svc.predict(X_test)

    cm_svc += confusion_matrix(y_test, y_svc_pred, range(NUM_OF_CLF))
    cm_lin_svc += confusion_matrix(y_test, y_lin_svc_pred, range(NUM_OF_CLF))

    svc_score[i] = svc.score(X_test, y_test)
    lin_svc_score[i] =  lin_svc.score(X_test, y_test)

np.set_printoptions(precision=4, suppress=True)

cm_svc_normalized = cm_svc.astype('float') / cm_svc.sum(axis=1)[:, np.newaxis]
cm_lin_svc_normalized = cm_lin_svc.astype('float') / cm_lin_svc.sum(axis=1)[:, np.newaxis]

print 'Linear SVC:\n\tmean:', svc_score.mean(), '\n\tstd:', svc_score.std()
# print cm_svc_normalized
# print (cm_svc.T / cm_svc.sum(axis=1) ).T

print 'LinearSVC:\n\tmean:', lin_svc_score.mean(), '\n\tstd:', lin_svc_score.std()
# print cm_lin_svc_normalized
# print (cm_lin_svc.T / cm_lin_svc.sum(axis=1) ).T

# Write file to csv file
# with open(RESULT_PATH + 'confusion_matrix_' + TYPE + '.csv', 'w') as fp:
#     a = csv.writer(fp, delimiter=',')
#     a.writerows(cm_svc_normalized)

# Write file to csv file with precision control
with open(RESULT_PATH + 'confusion_matrix_' + TYPE + '.csv', 'w') as fp:
    wr = csv.writer(fp, quoting=csv.QUOTE_NONE)
    for i in cm_svc_normalized:
        wr.writerow(['{:.4f}'.format(x) for x in i])
