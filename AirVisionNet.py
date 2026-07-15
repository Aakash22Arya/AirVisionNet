# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:22:05 2022

@author: User
"""

#Import the libraries
import zipfile
import os
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import load_model
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model
from keras.layers.convolutional import Conv2D
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle

import random

def image_generator(files, batch_size = 16, sz = (1024, 1024)):
  
  while True: 
    
    batch = np.random.choice(files, size = batch_size, replace = True)
    batch_inp = []
    batch_out = []
    for f in batch:
        inputdata = Image.open(f'/a/mainProject/PM_Night_Annoted_train/{f}',mode='r')
        inputdata = inputdata.resize(sz)
        inputdata = np.array(inputdata)
        batch_inp.append(inputdata)
        f1 = f.split('@')
        z1 = f1[1]
        z2 = f1[2]
        l = len(z2)
        outputdata1 = float(z1)/850
        outputdata2 = float(z2[0:l-4])/1000
        # outputdata = np.array(outputdata)
        outputdata = np.array([outputdata1, outputdata2])
        batch_out.append(outputdata)

    batch_inp = np.array(batch_inp)/255
    batch_out = np.array(batch_out)
    
    yield (batch_inp, batch_out)
    
batch_size = 16

all_files = os.listdir('/a/mainProject/PM_Night_Annoted_train/')
random.shuffle(all_files)
# Number of folds
k = 5
fold_size = len(all_files) // k

# Split the files into k folds
folds = [all_files[i:i + fold_size] for i in range(0, len(all_files), fold_size)]

# Take the second fold for testing
test_files = folds[0]

# Combine the other folds for training
train_files = [file for fold in folds[1:] for file in fold]


train_generator = image_generator(train_files, batch_size)
test_generator  = image_generator(test_files, batch_size)

def dense_block(ip):
  concatenated_inputs = ip
  for i in range(3):
    conv1 = tf.keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(concatenated_inputs)
    x = tf.keras.layers.BatchNormalization()(conv1)
    concatenated_inputs = tf.keras.layers.Concatenate()([concatenated_inputs, x])
  return concatenated_inputs

def resnet_block(ip,n,k):
  factor = 64
  for l in range(n):
    conv1 = tf.keras.layers.Conv2D(64*k, 1, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(ip)
    ip = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = tf.keras.layers.Conv2D(64*k, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(ip)
    ip = tf.keras.layers.BatchNormalization()(conv2)
    conv3 = tf.keras.layers.Conv2D(256*k, 1, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(ip)
    ip = tf.keras.layers.BatchNormalization()(conv3)
  return ip

def unet_plus(ip):
  I1 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(ip)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(I1)
  up1 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(pool1)
  y1 = tf.keras.layers.Concatenate(axis = -1)([I1,up1])
  I2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(y1)
  

  T1 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(pool1)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(T1)
  up2 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(pool2)
  y2 = tf.keras.layers.Concatenate( axis = -1)([pool1,up2])
  up3 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(y2)
  y3 = tf.keras.layers.Concatenate( axis = -1)([I2,up3])
  I3 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(y3)


  T2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(pool2)
  pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(T2)
  up4 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(pool3)
  y4 = tf.keras.layers.Concatenate( axis = -1)([pool2,up4])
  up5 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(y4)
  y5 = tf.keras.layers.Concatenate(axis = -1)([y2,up5])
  up6 = tf.keras.layers.Conv2DTranspose(3, 2, strides=[2, 2], activation = 'relu',padding="same", kernel_initializer='he_normal')(y5)
  y6 = tf.keras.layers.Concatenate( axis = -1)([I3,up6])
  I4 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(y6)

  return I4 

def pdarknet(input_size = (1024,1024,3)):
  inputs = Input(input_size)
  conv1 = tf.keras.layers.Conv2D(3, 7, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(inputs)
  bn1 = tf.keras.layers.BatchNormalization()(conv1)

  ## High resolution network
  conv2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(bn1)
  bn2 = tf.keras.layers.BatchNormalization()(conv2)
  max1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid")(bn2)

  conv21 = tf.keras.layers.Conv2D(3, 1, activation = 'relu', padding = 'same', strides = 1, dilation_rate = 1, kernel_initializer = 'he_normal')(max1)
  bn21 = tf.keras.layers.BatchNormalization()(conv21)
  conv22 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, dilation_rate = 2, kernel_initializer = 'he_normal')(max1)
  bn22 = tf.keras.layers.BatchNormalization()(conv22)

  diff1 = tf.keras.layers.Subtract()([bn21, bn22])
  temp1 = tf.keras.activations.sigmoid(diff1)
  bn_diff1 = tf.keras.layers.BatchNormalization()(temp1)
  xc = tf.keras.layers.Multiply()([max1, bn_diff1])
  max11 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(xc)
  conv1_m1 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(max11)
  conv1_m2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(max11)
  conv1_m = tf.keras.layers.Multiply()([conv1_m1, conv1_m2])

  diff2 = tf.keras.layers.Subtract()([bn22, bn21])
  temp2 = tf.keras.activations.sigmoid(diff2)
  bn_diff2 = tf.keras.layers.BatchNormalization()(temp2)
  xc_com = tf.keras.layers.Subtract()([max1, bn_diff2]) 
  max22 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(xc_com)
  conv2_m1 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(max22)
  conv2_m2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(max22)
  conv2_m = tf.keras.layers.Multiply()([conv2_m1, conv2_m2])

  Hig_res_out = tf.keras.layers.Concatenate(axis=-1)([conv1_m,conv2_m])

  ## Medium resolution network
  conv3 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(bn1)
  bn3 = tf.keras.layers.BatchNormalization()(conv3)
  max2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(4,4), padding="valid")(bn3)

  Med_res_out = unet_plus(max2)
  #Med_res_out = unet(max2)

  ## Low resolution network
  conv4 = tf.keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(bn1)
  bn4 = tf.keras.layers.BatchNormalization()(conv4)
  max3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(8,8), padding="valid")(bn4)

  D1 = dense_block(max3)
  D2 = dense_block(D1)
  D3 = dense_block(D2)

  Low_res_out = tf.keras.layers.Conv2DTranspose(16, 2, strides=[2, 2], padding="same", kernel_initializer='he_normal')(D3)
  concat = tf.keras.layers.Concatenate(axis=-1)([Hig_res_out,Med_res_out,Low_res_out])

  ## PM computation
  bn5 = tf.keras.layers.BatchNormalization()(concat)
  C1 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn5)
  concat1 = tf.keras.layers.Concatenate(axis=-1)([concat,C1])  
  bn6 = tf.keras.layers.BatchNormalization()(concat1)
  C2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn6)
  max4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")(C2)


  bn7 = tf.keras.layers.BatchNormalization()(max4)
  C3 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn7)
  concat2 = tf.keras.layers.Concatenate(axis=-1)([max4,C3])  
  bn8 = tf.keras.layers.BatchNormalization()(concat2)
  C4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn8)
  max5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")(C4)


  bn9 = tf.keras.layers.BatchNormalization()(max5)
  C5 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn9)
  concat3 = tf.keras.layers.Concatenate(axis=-1)([max5,C5])  
  bn10 = tf.keras.layers.BatchNormalization()(concat3)
  C6 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation = 'relu', dilation_rate=1)(bn10)
  max6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")(C6)

  bn10 = tf.keras.layers.BatchNormalization()(max6)
  C7 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', strides = 1, kernel_initializer = 'he_normal')(bn10)
  f = tf.keras.layers.GlobalAveragePooling2D()(C7)

  ## Model Training
  model = Model(inputs = inputs, outputs= f)
  model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'mse', metrics = [tf.keras.metrics.MeanAbsoluteError()])
  # model.summary()
  return model

# network = pdarknet()

batch_size = 16
train_steps = len(train_files) //batch_size
#train_steps = len(train_files)
test_steps = len(test_files) //batch_size
#test_steps = len(test_files)
model = pdarknet()


model.load_weights('/a/mainProject/MainModel_2.h5')

batch_size = 16
train_steps = len(train_files) //batch_size
test_steps = len(test_files) //batch_size
EPOCHS = 1000
STEPS_PER_EPOCH = train_steps
model_checkpoint = ModelCheckpoint("MainModel_2.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
model.fit(train_generator, initial_epoch = 0, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = test_generator, validation_steps = test_steps,callbacks=[model_checkpoint])

