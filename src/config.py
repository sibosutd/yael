'''
Configuration file
'''
import os, sys, pickle
import numpy as np
from numpy import genfromtxt
from sklearn import svm, preprocessing

# modify path variable according to your directory
BASE_PATH = '/home/sibo/Documents/Projects/'
LIBRARY_PATH = BASE_PATH + 'yael/yael_v438/'
MODEL_PATH = BASE_PATH + 'yael/model/'

# DATASET_PATH = BASE_PATH + 'multimodal_sensors/sensor_data/data_txt/'

sys.path.append(LIBRARY_PATH)
from yael import ynumpy
