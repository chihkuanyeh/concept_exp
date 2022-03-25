# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper file to run the discover concept algorithm in the AwA dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import pickle
from absl import app
import keras
import keras.backend as K
from keras.layers import Dense
from sklearn.cluster import KMeans
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D
from keras.models import load_model
from keras.optimizers import SGD, Adam
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from inception_v3 import conv2d_bn
from PIL import Image
import cv2
backend = keras.backend
layers = keras.layers
models = keras.models

if backend.image_data_format() == 'channels_first':
  channel_axis = 1
else:
  channel_axis = 3


def load_model():
  K.set_learning_phase(0)
  with open('data/classes.pickle', 'rb') as handle:
    classes = pickle.load(handle)
  print(classes)
  """Loads the pretrained model."""
  model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                      weights='imagenet')
  model.layers.pop()
  for layer in model.layers:
    layer.trainable = False
  last = model.layers[-1].output
  dense1 = Dense(1024, activation='relu', name='concept1')
  dense2 = Dense(1024, activation='relu', name='concept2')
  fc1 = dense1(last)
  fc2 = dense2(fc1)
  predict = Dense(len(classes), name='output')
  logits = predict(fc2)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))

    return loss
  finetuned_model = Model(model.input, logits)
  finetuned_model.compile(
      optimizer=SGD(lr=0.01),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  finetuned_model.classes = classes
  finetuned_model.load_weights('data/inception_final.h5')
  feature_dense_model = Model(model.input, fc1)
  fc1_input = Input(shape=(1024,))
  fc2_temp = dense2(fc1_input)
  logits_temp = predict(fc2_temp)
  fc_model = Model(fc1_input, logits_temp)
  fc_model.compile(
      optimizer=SGD(lr=0.01),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  for layer in finetuned_model.layers:
    layer.trainable = False
  return finetuned_model, feature_dense_model, fc_model, dense2, predict

def load_model_stm(layer_num):
  with open('data/classes.pickle', 'rb') as handle:
    classes = pickle.load(handle)
  """Loads the pretrained model."""
  model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                      weights='imagenet')
  model.layers.pop()
  for layer in model.layers:
    layer.trainable = False
  last = model.layers[-1].output
  dense1 = Dense(1024, activation='relu', name='concept1')
  dense2 = Dense(1024, activation='relu', name='concept2')
  fc1 = dense1(last)
  fc2 = dense2(fc1)
  predict = Dense(len(classes), name='output')
  logits = predict(fc2)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))

    return loss
  finetuned_model = Model(model.input, logits)
  finetuned_model.compile(
      optimizer=SGD(lr=0.00),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  #finetuned_model.classes = classes
  finetuned_model.load_weights('data/inception_final.h5')
  #print(finetuned_model.summary())
  weight_list = []
  for id in range(layer_num,0,1):
    weight_list.append(finetuned_model.layers[id].get_weights())
  feature_layer = finetuned_model.layers[layer_num].input
  feature_model = Model(model.input, feature_layer)
  input1 = Input(shape=(8,8,2048))

  # mixed 9: 8 x 8 x 2048
  x = Lambda(lambda x:x)(input1)
  for i in range(1):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        # Classification block
  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

  fc1_2 = dense1(x)
  fc2_2 = dense2(fc1_2)
  logits_2 = predict(fc2_2)

  predict_model = Model(input1, logits_2)
  #print(predict_model.summary())
  for count,layer in enumerate(predict_model.layers[2:]):
    print(count)
    if weight_list[count] != None:
      layer.set_weights(weight_list[count])

  predict_model.compile(
      optimizer=SGD(lr=0.00),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  for layer in feature_model.layers:
    layer.trainable = False
  for layer in predict_model.layers:
    layer.trainable = False
    if isinstance(layer, keras.layers.normalization.BatchNormalization):
      layer._per_input_updates = {}
  return feature_model, predict_model

def load_model_stm_small_new(layer_num, trainable = True):
  with open('data/classes.pickle', 'rb') as handle:
    classes = pickle.load(handle)
  """Loads the pretrained model."""
  model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                      weights='imagenet')
  model.layers.pop()
  print(model.summary())
  for layer in model.layers:
    layer.trainable = True
  last = model.layers[layer_num].output
  last2 = model.layers[-84].output
  #print(model.summary())

  
  #conv2d = conv2d_bn(branch3x3dbl, 384, 3, 3)
  dense1 = Dense(1024, activation='relu', name='concept1')
  dense2 = Dense(1024, activation='relu', name='concept2')
  last_x = layers.GlobalAveragePooling2D(name='avg_pool')(last2)
  fc1 = dense1(last_x)
  fc2 = dense2(fc1)
  predict = Dense(len(classes), name='output')
  logits = predict(fc2)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))

    return loss

  feature_model = Model(model.input, last)
  finetuned_model = Model(model.input, logits)


  #finetuned_model.classes = classes
  #finetuned_model.load_weights('data/inception_final.h5')
  print(finetuned_model.summary())



  #feature_layer = finetuned_model.layers[layer_num].input
  #feature_model = Model(model.input, feature_layer)

  input1 = Input(shape=(35,35,288))
  '''
  # mixed 0: 35 x 35 x 256
  branch1x1 = conv2d_bn(input1, 64, 1, 1)

  branch5x5 = conv2d_bn(input1, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(input1, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(input1)
  branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
  x = layers.concatenate(
      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
      axis=channel_axis,
      name='mixed0')


  # mixed 1: 35 x 35 x 288
  branch1x1 = conv2d_bn(input1, 64, 1, 1)

  branch5x5 = conv2d_bn(input1, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(input1, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(input1)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = layers.concatenate(
      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
      axis=channel_axis,
      name='mixed1')
    '''
  # mixed 2: 35 x 35 x 288
  branch1x1 = conv2d_bn(input1, 64, 1, 1)

  branch5x5 = conv2d_bn(input1, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(input1, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(input1)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = layers.concatenate(
      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
      axis=channel_axis,
      name='mixed2')

  # mixed 3: 17 x 17 x 768
  branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(
      branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.concatenate(
      [branch3x3, branch3x3dbl, branch_pool],
      axis=channel_axis,
      name='mixed3')

  # mixed 4: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 128, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 128, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = layers.concatenate(
      [branch1x1, branch7x7, branch7x7dbl, branch_pool],
      axis=channel_axis,
      name='mixed4')

  # mixed 5, 6: 17 x 17 x 768
  for i in range(2):
      branch1x1 = conv2d_bn(x, 192, 1, 1)

      branch7x7 = conv2d_bn(x, 160, 1, 1)
      branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
      branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

      branch7x7dbl = conv2d_bn(x, 160, 1, 1)
      branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
      branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
      branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
      branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

      branch_pool = layers.AveragePooling2D(
          (3, 3), strides=(1, 1), padding='same')(x)
      branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
      x = layers.concatenate(
          [branch1x1, branch7x7, branch7x7dbl, branch_pool],
          axis=channel_axis,
          name='mixed' + str(5 + i))

  # mixed 7: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 192, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 192, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = layers.concatenate(
      [branch1x1, branch7x7, branch7x7dbl, branch_pool],
      axis=channel_axis,
      name='mixed7')
  '''
  # mixed 9: 8 x 8 x 2048
  x = Lambda(lambda x:x)(input1)
  for i in range(1):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        # Classification block
  '''
  #input1_conv = conv1(input1)
  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

  fc1_2 = dense1(x)
  fc2_2 = dense2(fc1_2)
  logits_2 = predict(fc2_2)

  finetuned_model.load_weights('small_inception_weight.h5')
  predict_model = Model(input1, logits_2)
  print(finetuned_model.layers[41].get_weights)
  #print(predict_model.summary())

  weight_list = []
  
  for id in range(64,len(finetuned_model.layers),1):
    weight_list.append(finetuned_model.layers[id].get_weights())
    #print(weight_list[-1][0].shape)
    #input()

  print(len(weight_list))

  for count,layer in enumerate(predict_model.layers[1:]):
    print(count)
    if weight_list[count] != None:
      layer.set_weights(weight_list[count])

  predict_model.compile(
      optimizer=SGD(lr=0.00),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])
  

  for layer in feature_model.layers:
    layer.trainable = trainable

  for count,layer in enumerate(predict_model.layers):
    if count >=60:
      layer.trainable = True
    else:
      layer.trainable = False
    #if isinstance(layer, keras.layers.normalization.BatchNormalization):
    #  layer._per_input_updates = {}

  finetuned_model.compile(
      optimizer=Adam(lr=0.001),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])
  return feature_model, predict_model, finetuned_model

def load_model_stm_small(layer_num, trainable = True):
  with open('data/classes.pickle', 'rb') as handle:
    classes = pickle.load(handle)
  """Loads the pretrained model."""
  model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                      weights='imagenet')
  model.layers.pop()
  print(model.summary())
  for layer in model.layers:
    layer.trainable = True
  last = model.layers[layer_num].output
  last2 = model.layers[-84].output
  #print(model.summary())

  
  #conv2d = conv2d_bn(branch3x3dbl, 384, 3, 3)
  dense1 = Dense(1024, activation='relu', name='concept1')
  dense2 = Dense(1024, activation='relu', name='concept2')
  last_x = layers.GlobalAveragePooling2D(name='avg_pool')(last2)
  fc1 = dense1(last_x)
  fc2 = dense2(fc1)
  predict = Dense(len(classes), name='output')
  logits = predict(fc2)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))

    return loss

  feature_model = Model(model.input, last)
  finetuned_model = Model(model.input, logits)

  '''
  #finetuned_model.classes = classes
  finetuned_model.load_weights('data/inception_final.h5')
  print(finetuned_model.summary())
  '''

  #feature_layer = finetuned_model.layers[layer_num].input
  #feature_model = Model(model.input, feature_layer)

  input1 = Input(shape=(17,17,768))
  for i in range(1):
        branch1x1 = conv2d_bn(input1, 192, 1, 1)

        branch7x7 = conv2d_bn(input1, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(input1, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(input1)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))
  for i in range(1):
    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
  '''
  # mixed 9: 8 x 8 x 2048
  x = Lambda(lambda x:x)(input1)
  for i in range(1):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        # Classification block
  '''
  #input1_conv = conv1(input1)
  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

  fc1_2 = dense1(x)
  fc2_2 = dense2(fc1_2)
  logits_2 = predict(fc2_2)

  predict_model = Model(input1, logits_2)
  #print(finetuned_model.summary())
  #print(predict_model.summary())

  weight_list = []
  for id in range(-68,0,1):
    weight_list.append(finetuned_model.layers[id].get_weights())
    #print(weight_list[-1][0].shape)
    #input()

  print(len(weight_list))

  for count,layer in enumerate(predict_model.layers[1:]):
    print(count)
    if weight_list[count] != None:
      layer.set_weights(weight_list[count])

  predict_model.compile(
      optimizer=SGD(lr=0.00),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])
  
  finetuned_model.load_weights('small_inception_weight.h5')
  for layer in feature_model.layers:
    layer.trainable = trainable

  for layer in predict_model.layers:
    layer.trainable = trainable
    #if isinstance(layer, keras.layers.normalization.BatchNormalization):
    #  layer._per_input_updates = {}

  finetuned_model.compile(
      optimizer=Adam(lr=0.001),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])
  return feature_model, predict_model, finetuned_model


def copy_save_image(f1,f2,a,b,da,db):
  # open the image
  Image1 = Image.open(f1)

  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((299,299), Image.ANTIALIAS)
  left = int(172.0/17*b) 
  right = left+127
  if right>=299:
    right = 299
  top = int(172.0/17*b) 
  bottom = top+127
  if bottom>=299:
    bottom = 299

  region = Image1copy.crop((left,top,right,bottom))
  region.save(f2)

def copy_save_image_early(f1,f2,a,b,da,db):
      # open the image
  Image1 = Image.open(f1)

  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((299,299), Image.ANTIALIAS)
  left = int((299-95)/35*b) 
  right = left+95
  if right>=299:
    right = 299
  top = int((299-95)/35*b) 
  bottom = top+95
  if bottom>=299:
    bottom = 299

  region = Image1copy.crop((left,top,right,bottom))
  region.save(f2)

def get_ace_concept_stm(cluster_new, predict, f_train):
  """Calculates ACE/TCAV concepts."""
  concept_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='concept_input')
  softmax_tcav = predict(concept_input)
  tcav_model = Model(inputs=concept_input, outputs=softmax_tcav)
  tcav_model.layers[-1].activation = None
  tcav_model.layers[-1].trainable = False
  tcav_model.layers[-2].trainable = False

  tcav_model.compile(
      loss='mean_squared_error',
      optimizer=SGD(lr=0.0),
      metrics=['binary_accuracy'])
  tcav_model.summary()

  n_cluster = cluster_new.shape[0]
  n_percluster = cluster_new.shape[1]
  print(cluster_new.shape)
  weight_ace = np.zeros((768, n_cluster))
  tcav_list_rand = np.zeros((50, 30))
  tcav_list_ace = np.zeros((50, n_cluster))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(cluster_new.reshape((-1, 768)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((768, 30))
  for i in range(30):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(cluster_new.reshape((-1, 768)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(50):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 50),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = np.mean(gradient_function([f_train])[0],axis=(1,2))

    grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, weight_ace)
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 1.0).astype(int)
    sig_list += (tcav_list_ace[j, :] < mean - std * 1.0).astype(int)
  top_k_index = np.array(sig_list).argsort()[::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace

def load_data(train_dir, size, batch_size, feature_model, predict_model, finetuned_model,
              pretrained=True, noise=0.):
  """Loads data and adding noise."""
  #crop_length = 149
  def random_crop(img, random_crop_size,x,y):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    im2 = np.zeros(img.shape)
    im2[y*dy:(y+dy), x*dx:(x+dx), :] = img[y*dy:(y+dy), x*dx:(x+dx), :]
    #x = np.random.randint(0, width - dx + 1)
    #y = np.random.randint(0, height - dy + 1)
    #return img[y*dy:(y*dy+dy), x*dx:(x*dx+dx), :]
    return im2

  def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        #batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        batch_crops = np.zeros((4*batch_x.shape[0], 299, 299, 3))
        batch_ys = np.tile(batch_y, (4, 1))  # repeat 'y' 5 times
        for i in range(batch_x.shape[0]):
          batch_crops[i*4] = random_crop(batch_x[i], (crop_length, crop_length),0,0)
          batch_crops[i*4+1] = random_crop(batch_x[i], (crop_length, crop_length),0,1)
          batch_crops[i*4+2] = random_crop(batch_x[i], (crop_length, crop_length),1,0)
          batch_crops[i*4+3] = random_crop(batch_x[i], (crop_length, crop_length),1,1)
        #batch_crops_resize = cv2.resize(batch_crops, dsize=(299,299), interpolation=cv2.INTER_CUBIC)
        #print('success')
        yield (batch_crops, batch_ys)

        #for i in range(batch_x.shape[0]):
        #    batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length),x*crop_length,y*crop_length)
        #yield (batch_crops, batch_y)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))
    return loss

  def rand_noise(img):
    img_noisy = img + np.random.normal(scale=noise, size=img.shape)
    return img_noisy

  if not pretrained:
    #print(K.tensorflow_backend._get_available_gpus())
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    gen_noisy = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, preprocessing_function=rand_noise)
    aug = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    batches = aug.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        subset='training')
    batches_fix_train = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='training')
    batches_fix_val = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='validation')

    classes = list(iter(batches.class_indices))
    for c in batches.class_indices:
      classes[batches.class_indices[c]] = c
    num_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
    num_train_steps = math.floor(num_train_samples * 0.9 / batch_size) 
    num_valid_steps = math.floor(num_train_samples * 0.1 / batch_size)
    #train_crops= crop_generator(batches_fix_train, 149)
    #del finetuned_model
    finetuned_model.load_weights('small_inception_weight.h5')

    print(
    finetuned_model.evaluate_generator(
        batches_fix_val,
        workers=1,
        verbose=True,
        use_multiprocessing=False))
    f_train = feature_model.predict_generator(
        batches_fix_train,
        steps=num_train_steps,
        workers=10,
        use_multiprocessing=False)
    
    np.save('awa_data/f_train_small.npy', f_train)
    print(f_train.shape)
    print(num_valid_steps)
    print(batch_size)
    f_val = feature_model.predict_generator(
        batches_fix_val,
        steps=num_valid_steps,
        workers=10,
        use_multiprocessing=False)
    y_train = batches_fix_train.classes
    y_val = batches_fix_val.classes
    y_train_logit = tf.keras.utils.to_categorical(
        y_train[:f_train.shape[0]], num_classes=50)
    y_val_logit = tf.keras.utils.to_categorical(
        y_val[:f_val.shape[0]],
        num_classes=50,
    )
    np.save('awa_data/y_train_logit.npy', y_train_logit)
    np.save('awa_data/y_val_logit.npy', y_val_logit)
    np.save('awa_data/y_train.npy', y_train)
    np.save('awa_data/y_val.npy', y_val)

    np.save('awa_data/f_val_small.npy', f_val)
    with open('awa_data/classes.pickle', 'wb') as handle:
      pickle.dump(classes, handle, pickle.HIGHEST_PROTOCOL)
  else:
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    gen_noisy = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, preprocessing_function=rand_noise)
    aug = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    batches_fix_train = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='training')

    batches = aug.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        subset='training')
    batches_fix_val = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='validation')
    filename = batches_fix_train.filenames
    np.save('awa_data/file_name',filename)
    print(filename[:100])
    y_train_logit = np.load('awa_data/y_train_logit.npy')
    y_val_logit = np.load('awa_data/y_val_logit.npy')
    y_train = np.load('awa_data/y_train.npy')
    y_val = np.load('awa_data/y_val.npy')
    f_train = np.load('awa_data/f_train_small.npy')
    f_val = np.load('awa_data/f_val_small.npy')
  #print(predict_model.evaluate(
  #        f_train, y_train_logit,
  #        verbose=True))

  return y_train_logit, y_val_logit, y_train, y_val, \
      f_train, 0, f_val, batches, batches_fix_val


def load_data_new(train_dir, size, batch_size, feature_model, predict_model, finetuned_model,
              pretrained=True, noise=0.):
  """Loads data and adding noise."""
  #crop_length = 149
  def random_crop(img, random_crop_size,x,y):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    im2 = np.zeros(img.shape)
    im2[y*dy:(y+dy), x*dx:(x+dx), :] = img[y*dy:(y+dy), x*dx:(x+dx), :]
    #x = np.random.randint(0, width - dx + 1)
    #y = np.random.randint(0, height - dy + 1)
    #return img[y*dy:(y*dy+dy), x*dx:(x*dx+dx), :]
    return im2

  def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        #batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        batch_crops = np.zeros((4*batch_x.shape[0], 299, 299, 3))
        batch_ys = np.tile(batch_y, (4, 1))  # repeat 'y' 5 times
        for i in range(batch_x.shape[0]):
          batch_crops[i*4] = random_crop(batch_x[i], (crop_length, crop_length),0,0)
          batch_crops[i*4+1] = random_crop(batch_x[i], (crop_length, crop_length),0,1)
          batch_crops[i*4+2] = random_crop(batch_x[i], (crop_length, crop_length),1,0)
          batch_crops[i*4+3] = random_crop(batch_x[i], (crop_length, crop_length),1,1)
        #batch_crops_resize = cv2.resize(batch_crops, dsize=(299,299), interpolation=cv2.INTER_CUBIC)
        #print('success')
        yield (batch_crops, batch_ys)

        #for i in range(batch_x.shape[0]):
        #    batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length),x*crop_length,y*crop_length)
        #yield (batch_crops, batch_y)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=y_true, logits=y_pred))
    return loss

  def rand_noise(img):
    img_noisy = img + np.random.normal(scale=noise, size=img.shape)
    return img_noisy

  if not pretrained:
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    gen_noisy = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, preprocessing_function=rand_noise)
    aug = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    batches = aug.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        subset='training')
    batches_fix_train = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='training')
    batches_fix_val = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='validation')

    classes = list(iter(batches.class_indices))
    for c in batches.class_indices:
      classes[batches.class_indices[c]] = c
    num_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
    num_train_steps = math.floor(num_train_samples * 0.9 / batch_size) 
    num_valid_steps = math.floor(num_train_samples * 0.1 / batch_size)
    #train_crops= crop_generator(batches_fix_train, 149)
    #del finetuned_model
    finetuned_model.load_weights('small_inception_weight.h5')

    print(
    finetuned_model.evaluate_generator(
        batches_fix_val,
        workers=1,
        verbose=True,
        use_multiprocessing=False))
    f_train = feature_model.predict_generator(
        batches_fix_train,
        steps=num_train_steps,
        workers=10,
        use_multiprocessing=False)
    
    np.save('awa_data/f_train_small_new.npy', f_train)
    print(f_train.shape)
    print(num_valid_steps)
    print(batch_size)
    f_val = feature_model.predict_generator(
        batches_fix_val,
        steps=num_valid_steps,
        workers=10,
        use_multiprocessing=False)
    y_train = batches_fix_train.classes
    y_val = batches_fix_val.classes
    y_train_logit = tf.keras.utils.to_categorical(
        y_train[:f_train.shape[0]], num_classes=50)
    y_val_logit = tf.keras.utils.to_categorical(
        y_val[:f_val.shape[0]],
        num_classes=50,
    )
    np.save('awa_data/y_train_logit_new.npy', y_train_logit)
    np.save('awa_data/y_val_logit_new.npy', y_val_logit)
    np.save('awa_data/y_train_new.npy', y_train)
    np.save('awa_data/y_val_new.npy', y_val)

    np.save('awa_data/f_val_small_new.npy', f_val)
    with open('awa_data/classes.pickle', 'wb') as handle:
      pickle.dump(classes, handle, pickle.HIGHEST_PROTOCOL)
  else:
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    gen_noisy = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, preprocessing_function=rand_noise)
    aug = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    batches_fix_train = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='training')

    batches = aug.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        subset='training')
    batches_fix_val = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='validation')
    filename = batches_fix_train.filenames
    np.save('awa_data/file_name',filename)
    print(filename[:100])
    y_train_logit = np.load('awa_data/y_train_logit_new.npy')
    y_val_logit = np.load('awa_data/y_val_logit_new.npy')
    y_train = np.load('awa_data/y_train_new.npy')
    y_val = np.load('awa_data/y_val_new.npy')
    f_train = np.load('awa_data/f_train_small_new.npy')
    f_val = np.load('awa_data/f_val_small_new.npy')
  '''
  print(predict_model.evaluate(
          f_train, y_train_logit,
          verbose=True))
  '''

  return y_train_logit, y_val_logit, y_train, y_val, \
      f_train, 0, f_val, batches, batches_fix_val


def target_category_loss(x, category_index, nb_classes):
  return x* K.one_hot([category_index], nb_classes)


def get_ace_concept(concept_arraynew_active, dense2, predict, f_train,
                    concepts_to_select):
  """Calculates the ACE concepts."""
  concept_input = Input(shape=(1024,), name='concept_input')
  fc2_tcav = dense2(concept_input)
  softmax_tcav = predict(fc2_tcav)
  tcav_model = Model(inputs=concept_input, outputs=softmax_tcav)
  tcav_model.layers[-1].activation = None
  tcav_model.layers[-1].trainable = False
  tcav_model.layers[-2].trainable = False
  tcav_model.compile(
      loss='mean_squared_error',
      optimizer=SGD(lr=0.0),
      metrics=['binary_accuracy'])
  tcav_model.summary()

  n_cluster = concept_arraynew_active.shape[0]
  n_percluster = concept_arraynew_active.shape[1]
  print(concept_arraynew_active.shape)
  weight_ace = np.zeros((1024, n_cluster))
  tcav_list_rand = np.zeros((50, 200))
  tcav_list_ace = np.zeros((50, 134))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 1024)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((1024, 200))
  for i in range(200):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 1024)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(50):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 50),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = gradient_function([f_train])[0]
    grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, weight_ace)
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 2.0).astype(int)
  top_k_index = np.array(sig_list).argsort()[-1 * concepts_to_select:][::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace[:, top_k_index]


def get_pca_concept(f_train, concepts_to_select):
  pca = PCA()
  pca.fit(f_train)
  weight_pca = np.zeros((768, concepts_to_select))
  for count, pc in enumerate(pca.components_):
    if count >= concepts_to_select:
      break
    weight_pca[:, count] = pc
  return weight_pca

def get_kmeans_concept(f_train, concepts_to_select):
  kmeans = KMeans(n_clusters=concepts_to_select, random_state=0).fit(f_train)
  weight_cluster = kmeans.cluster_centers_.T
  return weight_cluster


def main(_):
  return

if __name__ == '__main__':
  app.run(main)
