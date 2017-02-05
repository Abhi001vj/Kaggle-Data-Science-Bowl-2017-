# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
%matplotlib inline
import utils; reload(utils)
from utils import *

%matplotlib inline
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data


from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

path = "data/dogscatsreduxkernelsedition/"
model_path = path + 'models/'
# results_path
if not os.path.exists(model_path): os.mkdir(model_path)

% cd data/dogscatsreduxkernelsedition
% cd train
g  = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000): os.rename(shuf[i],'../valid/' + shuf[i])

% mkdir ../sample
% mkdir ../sample/train
% mkdir ../sample/valid

from shutil import copyfile

g  = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000): copyfile(shuf[i],'../sample/train/' + shuf[i])
% cd ../valid

g  = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000): copyfile(shuf[i],'../sample/valid/' + shuf[i])
% cd train
% mkdir cats
% mkdir dogs
% mv cat.*.jpg cats/
% mv dog.*.jpg dogs/

% cd valid
% mkdir cats
% mkdir dogs
% mv cat.*.jpg cats/
% mv dog.*.jpg dogs/

% cd sample/train
% mkdir cats
% mkdir dogs
% mv cat.*.jpg cats/
% mv dog.*.jpg dogs/

%cd sample/valid
% mkdir cats
% mkdir dogs
% mv cat.*.jpg cats/
% mv dog.*.jpg dogs/

% cd test
% mkdir unknown
% mv *.jpg unknown/

batch_size=100
#batch_size=4

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model


# vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)


vgg.model.save_weights(path+'models/ft1.h5')


batches, preds = vgg.test(path+'test', batch_size = batch_size)

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]


filenames = batches.filenames
save_array(path+'results/test_preds.dat',preds)
save_array(path+'results/filenames.dat',filenames)
preds = load_array('results/test_preds.dat')
filenames = load_array('results/filenames.dat')
isdog = preds[:,1]
ids =[int(f[8:f.find('.')]) for f in filenames]
subm = np.stack([ids,isdog], axis = 1)

np.savetxt('data/dogscatsreduxkernelsedition/subm98.csv' subm, fmt='%d,%.5f', header='id,label', comments='')  

from IPython.display import FileLink
FileLink('data/dogscatsreduxkernelsedition/subm98.csv')
# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)
