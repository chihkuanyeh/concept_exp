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

"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model


from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from skimage.segmentation import felzenszwalb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from PIL import Image
from matplotlib import cm
seed(0)
#tf.set_random_seed(0)
tf.random.set_seed(0)
batch_size = 64

def copy_save_image(x,f1,f2,a,b):

  # open the image
  Image1 = Image.fromarray(x.astype('uint8'))
  Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
  left = 32*b
  right = left+116
  top = 32*a
  bottom = top+116

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  old_size = (116,116)
  new_size = (118,118)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  new_im.save(f2)

def copy_save_image_all(x,f1,f2,a,b):

  # open the image
  Image1 = Image.fromarray(x.astype('uint8'))
  old_size = (240,240)
  new_size = (244,244)
  new_im = Image.new("RGB", new_size)
  new_im.paste(Image1, (2,2))
  new_im.save(f2)
  '''
  Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
  left = 32*b
  right = left+116
  top = 32*a
  bottom = top+116

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  old_size = (116,116)
  new_size = (118,118)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  '''
  #Image1.save(f2)

def load_xyconcept(n, pretrain):
  """Loads data and create label for toy dataset."""
  concept = np.load('toy_data/concept_data.npy')
  y = np.zeros((n, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])
  if not pretrain:
    x = np.load('../x_data.npy') / 255.0
    return x, y, concept
  return 0, y, concept


def target_category_loss(x, category_index, nb_classes):
  return x * K.one_hot([category_index], nb_classes)


def load_model_stm_new(x_train, y_train, x_val, y_val, width=240, \
               height=240, channel=3, pretrain=True):
  """Loads pretrain model or train one."""
  input1 = Input(
      shape=(
          width,
          height,
          channel,
      ), name='concat_input')
  conv1 = Conv2D(64, kernel_size=5, activation='relu')
  conv2 = Conv2D(64, kernel_size=5, activation='relu')
  conv3 = Conv2D(64, kernel_size=5, activation='relu')
  dense1 = Dense(200, activation='relu')
  predict = Dense(15, activation='sigmoid')
  conv1o = conv1(input1)
  pool1 = MaxPooling2D(pool_size=(4, 4))(conv1o)
  conv2o = conv2(pool1)
  pool2 = MaxPooling2D(pool_size=(4, 4))(conv2o)
  conv3o = conv3(pool2)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3o)
  pool3f = Flatten()(pool3)
  fc1 = dense1(pool3f)
  sigmoid1 = predict(fc1)

  pool3_2 = Input(shape=(4,4,64), name='concat_input')  
  pool3f_2 = Flatten()(pool3_2)
  fc1_2 = dense1(pool3f_2)
  #fc2_2 = dense2(fc1_2)
  sigmoid1_2 = predict(fc1_2)

  mlp = Model(input1, sigmoid1)
  mlp.compile(
      loss='binary_crossentropy',
      optimizer=Adam(lr=0.001),
      metrics=['binary_accuracy'])

  if pretrain:
    mlp.load_weights('toy_data/conv_toy.h5')
  else:
    _ = mlp.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
        verbose=1,
        validation_data=(x_val, y_val))
    mlp.save_weights('toy_data/conv_toy.h5')

  for layer in mlp.layers:
    layer.trainable = False

  feature_model = Model(input1, pool3)
  predict_model  = Model(pool3_2, sigmoid1_2)

  return feature_model, predict_model

def get_ace_concept(concept_arraynew_active, dense2, predict, f_train,
                    n_concept):
  """Calculates ACE/TCAV concepts."""
  concept_input = Input(shape=(200,), name='concept_input')
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
  weight_ace = np.zeros((200, n_cluster))
  tcav_list_rand = np.zeros((15, 200))
  tcav_list_ace = np.zeros((15, n_cluster))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 200)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((200, 200))
  for i in range(200):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 200)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(15):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 15),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = gradient_function([f_train])[0]
    grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, weight_ace)
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 1.0).astype(int)
  top_k_index = np.array(sig_list).argsort()[-1 * n_concept:][::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace[:, top_k_index]

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
  weight_ace = np.zeros((64, n_cluster))
  tcav_list_rand = np.zeros((15, 300))
  tcav_list_ace = np.zeros((15, n_cluster))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(cluster_new.reshape((-1, 64)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((64, 300))
  for i in range(300):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(cluster_new.reshape((-1, 64)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(15):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 15),
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
  top_k_index = np.array(sig_list).argsort()[-1 * 15:][::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace[:, top_k_index]

def get_ace_concept_stm_2(cluster_new, predict, f_train):
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
  print(cluster_new.shape)
  n_cluster = 15
  tcav_list_ace = np.zeros((15, n_cluster))
  sig_list = np.zeros(n_cluster)
  for j in range(15):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 15),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = np.reshape(gradient_function([f_train])[0],(-1,1024))

    #grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, cluster_new)
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    sig_list += (tcav_list_ace[j, :] > 0.7).astype(int)
    '''
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 1.0).astype(int)
    sig_list += (tcav_list_ace[j, :] < mean - std * 1.0).astype(int)
    '''
  top_k_index = np.array(sig_list).argsort()[-1 * 15:][::-1]
  print(sig_list)
  print(top_k_index)
  return cluster_new[:, top_k_index]

def get_pca_concept(f_train):
  pca = PCA()
  pca.fit(f_train)
  weight_pca = np.zeros((64, 15))
  for count, pc in enumerate(pca.components_):
    if count>14:
      break
    weight_pca[:, count] = pc
  return weight_pca

def get_kmeans_concept(f_train, n_concept):
  kmeans = KMeans(n_clusters=n_concept, random_state=0).fit(f_train)
  weight_cluster = kmeans.cluster_centers_.T
  return weight_cluster

def create_dataset(n_sample=60000):
  """Creates toy dataset and save to disk."""
  concept = np.reshape(np.random.randint(2, size=15 * n_sample),
                       (-1, 15)).astype(np.bool_)
  concept[:15, :15] = np.eye(15)
  fig = Figure(figsize=(2.4, 2.4))
  canvas = FigureCanvas(fig)
  axes = fig.gca()
  axes.set_xlim([0, 10])
  axes.set_ylim([0, 10])
  axes.axis('off')
  width, height = fig.get_size_inches() * fig.get_dpi()
  width = int(width)
  height = int(height)
  print(width)
  location = [(1.3, 1.3), (3.3, 1.3), (5.3, 1.3), (7.3, 1.3), (9.3, 1.3),
              (1.3, 3.3), (3.3, 3.3), (5.3, 3.3), (7.3, 2.3), (9.3, 3.3),
              (1.3, 5.3), (3.3, 5.3), (5.3, 5.3), (7.3, 5.3), (9.3, 5.3),
              (1.3, 7.3), (3.3, 7.3), (5.3, 7.3), (7.3, 7.3), (9.3, 7.3),
              (1.3, 9.3), (3.3, 9.3), (5.3, 9.3), (7.3, 9.3), (9.3, 9.3)]
  location_bool = np.zeros(25)
  x = np.zeros((n_sample, width, height, 3))
  color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']

  for i in range(n_sample):
    location_bool = np.zeros(25)
    if i % 10 == 0:
      print('{} images are created'.format(i))
    if concept[i, 5] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'x',
          color=color_array[np.random.randint(100) % 7],
          markersize=10,
          mew=4,
          ms=8)
    if concept[i, 6] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '3',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 7] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          's',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 8] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'p',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 9] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '_',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 10] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 11] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 12] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          11,
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 13] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'o',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 14] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '.',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 0] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '+',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 1] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '1',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 2] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '*',
          color=color_array[np.random.randint(100) % 7],
          markersize=30,
          mew=3,
          ms=5)
    if concept[i, 3] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '<',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 4] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'h',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    canvas.draw()
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3)

    x[i, :, :, :] = image
    fig = Figure(figsize=(2.4, 2.4))
    canvas = FigureCanvas(fig)
    axes = fig.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 10])
    axes.axis('off')
    # imgplot = plt.imshow(image)
    # plt.show()

  # create label by booling functions
  y = np.zeros((n_sample, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

  np.save('x_data.npy', x)
  np.save('y_data.npy', y)
  np.save('concept_data.npy', concept)

  return width, height


def get_groupacc(min_weight, f_train, f_val, concept,
                 n_concept, n_cluster, n0, verbose):
  """Gets the group accuracy for dicovered concepts."""
  #print(finetuned_model_pr.summary())
  #min_weight = finetuned_model_pr.layers[-5].get_weights()[0]
  
  loss_table = np.zeros((n_concept, 5))
  f_time_weight = np.matmul(f_train[:1000,:], min_weight)
  f_time_weight_val = np.matmul(f_val, min_weight)
  #f_time_weight[np.where(f_time_weight<0.8)]=0
  #f_time_weight_val[np.where(f_time_weight_val<0.8)]=0
  f_time_weight_m = np.max(f_time_weight,(1,2))
  f_time_weight_val_m = np.max(f_time_weight_val,(1,2))
  for count in range(n_concept):
    for count2 in range(5):

      #print('count 2 is {}'.format(count2))
      # count2 = max_cluster[count]
      mean0 = np.mean(
          f_time_weight_m[:,count][concept[:1000,count2] == 0]) * 100
      mean1 = np.mean(
          f_time_weight_m[:,count][concept[:1000,count2] == 1]) * 100

      if mean0 < mean1:
        pos = 1
      else:
        pos = -1
      best_err = 1e10
      best_bias = 0
      a = int((mean1 - mean0) / 10)
      if a == 0:
        a = pos

      
      for bias in range(int(mean0), int(mean1), a):
        if pos == 1:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  f_time_weight_m[:,count] >
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    f_time_weight_m[:,count] > bias / 100.))
            best_bias = bias
        else:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  f_time_weight_m[:,count] <
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    f_time_weight_m[:,count] < bias / 100.))
            best_bias = bias
      if pos == 1:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                f_time_weight_val_m[:,count] >
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  f_time_weight_val_m[:,count] > best_bias / 100.))
                /12000)
      else:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                f_time_weight_val_m[:,count] <
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  f_time_weight_val_m[:,count] < best_bias / 100.))
                /12000)
  print(np.amin(loss_table, axis=0))
  acc = np.mean(np.amin(loss_table, axis=0))
  print(acc)
  return acc

def get_groupacc_max(min_weight, f_train, f_val, concept,
                 n_concept, n_cluster, n0, verbose):
  """Gets the group accuracy for dicovered concepts."""
  #print(finetuned_model_pr.summary())
  #min_weight = finetuned_model_pr.layers[-5].get_weights()[0]

  loss_table = np.zeros((n_concept, 5))
  for count in range(n_concept):
    for count2 in range(5):
      #print('count 2 is {}'.format(count2))
      # count2 = max_cluster[count]
      mean0 = np.mean(
          np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2))[concept[:1000,count2] == 0]) * 100
      mean1 = np.mean(
          np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2))[concept[:1000,count2] == 1]) * 100

      if mean0 < mean1:
        pos = 1
      else:
        pos = -1
      best_err = 1e10
      best_bias = 0
      a = int((mean1 - mean0) / 10)
      if a == 0:
        a = pos
      for bias in range(int(mean0), int(mean1), a):
        if pos == 1:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) >
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) >
                    bias / 100.))
            best_bias = bias
        else:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) <
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) <
                    bias / 100.))
            best_bias = bias
      if pos == 1:
        ans = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.max(np.matmul(f_val, min_weight[:, count]),(1,2)) >
                best_bias / 100.)) / 12000
        loss_table[count, count2] = ans
        if verbose:
          print(ans)
      else:
        ans = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.max(np.matmul(f_val, min_weight[:, count]),(1,2)) <
                best_bias / 100.)) / 12000
        loss_table[count, count2] = ans
        if verbose:
          print(ans)
  print(np.amin(loss_table, axis=0))
  acc = np.mean(np.amin(loss_table, axis=0))
  print(acc)
  return acc

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
