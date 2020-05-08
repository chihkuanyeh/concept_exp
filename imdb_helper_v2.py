"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
import keras
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib
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
tf.random.set_seed(0)
batch_size = 64
# set parameters:
max_features = 9000
maxlen = 400
batch_size = 32
embedding_dims = 600
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 1

stop_word_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def show_sentence_2(x_train, n_instance, start_index, end_index):
  INDEX_FROM=3
  x_temp = x_train[n_instance]
  word_to_id = keras.datasets.imdb.get_word_index()
  word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
  word_to_id["<PAD>"] = 0
  word_to_id["<START>"] = 1
  word_to_id["<UNK>"] = 2
  word_to_id["<UNUSED>"] = 3

  id_to_word = {value:key for key,value in word_to_id.items()}
  ss = ' '.join(id_to_word[id] for id in x_temp[start_index:end_index] )
  print(ss)
  return ss

def show_sentence(x_train, n_instance, n_index, dict_count):
  INDEX_FROM=3
  x_temp = x_train[n_instance]
  word_to_id = keras.datasets.imdb.get_word_index()
  word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
  word_to_id["<PAD>"] = 0
  word_to_id["<START>"] = 1
  word_to_id["<UNK>"] = 2
  word_to_id["<UNUSED>"] = 3

  id_to_word = {value:key for key,value in word_to_id.items()}
  ss = ' '.join(id_to_word[id] for id in x_temp[n_index*2:n_index*2+9] )
  print(ss)
  for id in x_temp[n_index*2:n_index*2+9]:
    if id_to_word[id] not in dict_count:
      dict_count[id_to_word[id]]=1
    else:
      dict_count[id_to_word[id]]+=1
  return ss

def load_data(pretrain=False):
  print('Loading data...')
  if not pretrain:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = np.concatenate([x_train,x_test[:12500]],axis=0)
    y_train = np.concatenate([y_train,y_test[:12500]],axis=0)
    x_test = x_test[12500:]
    y_test = y_test[12500:]

    np.save('imdb_data/x_train.npy',x_train)
    np.save('imdb_data/y_train.npy',y_train)
    np.save('imdb_data/x_test.npy',x_test)
    np.save('imdb_data/y_test.npy',y_test)
  else:
    x_train = np.load('imdb_data/x_train.npy')
    x_test = np.load('imdb_data/x_test.npy')
    y_train = np.load('imdb_data/y_train.npy')
    y_test = np.load('imdb_data/y_test.npy')

  print(len(x_train), 'train sequences')
  print(len(x_test), 'test sequences')

  print('Pad sequences (samples x time)')
  x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
  x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)
  return x_train, x_test, y_train, y_test

def load_model_stm(x_train, y_train, x_val, y_val,pretrain=False):
  print('Build model...')
  print(x_train.shape)
  print(y_train.shape)
  #y_train = np.reshape(y_train,(25000,-1))
  #y_val = np.reshape(y_val,(25000,-1))
  #print(y_train.shape)
  input1 = Input(shape=(maxlen,), name='concat_input')
  embed1 = Embedding(max_features,
                      embedding_dims,
                      input_length=maxlen)
  conv1 = Conv1D(filters,
                  kernel_size,
                  padding='valid',
                  activation='relu',
                  strides=1)
  conv2 = Conv1D(250,
                  kernel_size,
                  padding='valid',
                  activation='relu',
                  strides=2)                  
  dense1 = Dense(hidden_dims, activation='relu')
  dense2 = Dense(1, activation='sigmoid')
  drop1 = Dropout(0.1, noise_shape=None, seed=None)
  drop2 = Dropout(0.1, noise_shape=None, seed=None)

  embed1o = drop1(embed1(input1))
  conv1o = (conv1(embed1o))
  #conv1o = (conv1(embed1o))
  conv2o = Flatten()(conv2(conv1o))
  conv2o_2 = Input(shape=(49000,), name='flatten_input')
  dense1o = drop2(dense1(conv2o))
  dense1o_2 = drop2(dense1(conv2o_2))
  dense2o = dense2(dense1o)
  dense2o_2 = dense2(dense1o_2)
  model = Model(input1, dense2o)
  print(model.summary())

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  if not pretrain:
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val))
    model.save_weights('imdb_data/imdb_new.h5')
  else:
    model.load_weights('imdb_data/imdb_new.h5')
    print(model.evaluate(x_val,y_val))

  feature_model = Model(input1, conv2o)
  predict_model  = Model(conv2o_2, dense2o_2)

  return model, feature_model, predict_model