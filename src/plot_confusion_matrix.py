'''
Plot confusion matrix
'''
__author__ = '1000892'
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

BASE_PATH = '/home/sibo/Documents/Projects/'
RESULT_PATH = BASE_PATH + 'yael/results/'
TYPE = 'video'

# read data from csv file
cm_normalized = genfromtxt(RESULT_PATH + 'confusion_matrix_' + TYPE + '.csv', delimiter=',')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()