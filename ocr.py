# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:47:53 2017

@author: betienne
"""
from data.char_data import load_data

import numpy as np

import tflearn
from tflearn.data_utils import to_categorical, shuffle 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

DATA_DIR = "" # Path to image samples
num_classes = 26

print("Loading data...")
X, y = load_data(DATA_DIR)
X, y = shuffle(X, y)
X = X.astype(np.float32)
y = to_categorical(y, num_classes)

print("Preprocessing images...")
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Redefinition of convnet_cifar10 network
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)

# Finetuning Softmax layer (Setting restore=False to not restore its weights)
softmax = fully_connected(network, num_classes, activation='softmax', restore=False)
regression = regression(softmax, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

model = tflearn.DNN(regression, checkpoint_path='letters',
                    max_checkpoints=3, tensorboard_verbose=0)

# Start training
model.fit(X, y, n_epoch=10, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='letters')

model.save('letters')




