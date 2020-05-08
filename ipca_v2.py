"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import app

import keras
from keras.activations import sigmoid
import keras.backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Layer
from keras.models import Model
from keras.layers import Flatten

from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np
from numpy import inf
from numpy.random import seed
from scipy.special import comb
import tensorflow as tf

seed(0)
tf.random.set_seed(0)

# global variables
init = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
batch_size = 256

step = 200
min_weight_arr = []
min_index_arr = []
concept_arr = {}


class Weight(Layer):
  """Simple Weight class."""

  def __init__(self, dim, **kwargs):
    self.dim = dim
    super(Weight, self).__init__(**kwargs)

  def build(self, input_shape):
    # creates a trainable weight variable for this layer.
    self.kernel = self.add_weight(
        name='proj', shape=self.dim, initializer=init, trainable=True)
    super(Weight, self).build(input_shape)

  def call(self, x):
    return self.kernel

  def compute_output_shape(self, input_shape):
    return self.dim

def given_loss(loss1):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return (tf.reduce_mean(input_tensor=loss1(y_true, y_pred)))
  return loss

def topic_loss(topic_prob_n, topic_vector_n, n_concept, f_input, loss1):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return (1.0*tf.reduce_mean(input_tensor=loss1(y_true, y_pred))
            - 10.0*tf.reduce_mean(input_tensor=(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=2,sorted=True).values))
            + 10.0*tf.reduce_mean(input_tensor=(K.dot(K.transpose(topic_vector_n), topic_vector_n) - np.eye(n_concept)))
            )
  return loss

def topic_loss_toy(topic_prob_n, topic_vector_n, n_concept, f_input, loss1, para = 1.0):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return (1.0*tf.reduce_mean(input_tensor=loss1(y_true, y_pred))\
            - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=32,sorted=True).values))
            + 0.1*tf.reduce_mean(input_tensor=(K.dot(K.transpose(topic_vector_n), topic_vector_n) - np.eye(n_concept)))
            )
  return loss

def topic_loss_nlp(topic_prob_n, topic_vector_n, n_concept, f_input, loss1, para = 1.0):
  """creates loss with regularization (for NLP)"""
  def loss(y_true, y_pred):
    return (tf.reduce_mean(input_tensor=loss1(y_true, y_pred))
            - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=16,sorted=True).values))
            + 0.1 *tf.reduce_mean(input_tensor=(K.dot(K.transpose(topic_vector_n), topic_vector_n) - np.eye(n_concept)))
            )
  return loss

def mean_sim(topic_prob_n,n_concept):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return 1*tf.reduce_mean(input_tensor=tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=32,sorted=True).values)
  return loss

def sample_binary(n_concept, n_sample, pp=0.2):
  """sample binary vectors for shapley calculation"""
  binary_matrix = np.zeros((n_sample,n_concept))
  remain = -1
  for i in range(n_sample):
    binary_matrix[i,:] = np.random.choice(2, n_concept, p=[1-pp, pp])

  return binary_matrix

def get_completeness(predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               topic_vector_init,
               verbose=False,
               epochs=20,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  f_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(f_input)
  topic_vector = Weight((f_train.shape[3], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=3, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[3]))(f_input)
  rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  pred = predict(rec_layer_2)
  topic_model_pr = Model(inputs=f_input, outputs=pred)
  topic_model_pr.layers[-1].trainable = True

  if load:
        topic_model_pr.load_weights(load)
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  topic_model_pr.layers[1].set_weights([topic_vector_init])
  topic_model_pr.layers[1].trainable = False
  topic_model_pr.layers[-1].trainable = False

  topic_model_pr.compile(
      loss=given_loss(loss1=loss1),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())

  topic_model_pr.fit(
          f_train,
          y_train,
          batch_size=128,
          epochs=epochs,
          validation_data=(f_val, y_val),
          verbose=verbose)
  return 0

def topic_model_new_crop(predict,
               f_train_crop,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  # f_input size (None, 8,8,2048)
  #input = Input(shape=(299,299,3), name='input')
  #f_input = get_feature(input)
  f_crop = Input(shape=(4, f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input_crop')
  f_crop_n =  Lambda(lambda x:K.l2_normalize(x,axis=(4)))(f_crop)

  f_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(f_input)
  # topic vector size (2048,n_concept)
  topic_vector = Weight((f_train.shape[3], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)

  topic_prob_crop_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_crop_n, topic_vector_n])
  # topic prob = batchsize * 8 * 8 * n_concept
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  #topic_prob_pos = Lambda(lambda x: K.maximum(x,-1000))(topic_prob)
  #print(K.sum(topic_prob, axis=3, keepdims=True))
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=3, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  # rec size is batchsize * 8 * 8 * 2048
  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[3]))(f_input)
  #rec = Lambda(lambda x:K.dot(x[0],K.transpose(x[1])))([topic_prob_pos, topic_vector])
  #scale_value = Weight((1,1,1,1))(f_input)
  #bias_value = Weight((1,1,1,2048))(f_input)
  #scaled_rec1 = Lambda(lambda x: x[0] * x[1])([rec, scale_value])
  #scaled_rec2 = Lambda(lambda x: x[0] + x[1])([scaled_rec1, bias_value])
  rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  #rec_layer_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(rec_layer)
  pred = predict(rec_layer_2)
  topic_model_pr = Model(inputs=[f_input,f_crop], outputs=pred)
  topic_model_pr.layers[-1].trainable = True
  #topic_model_pr.layers[1].trainable = False
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    # These depend on the optimizer class
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  # Later when you want to reset the optimizer
  #K.get_session().run(optimizer_reset)
  #print(metric1)
  metric1.append(mean_sim(topic_prob_crop_n, n_concept))
  topic_model_pr.compile(
      loss=topic_loss(topic_prob_crop_n, topic_vector_n,  n_concept, f_input, loss1=loss1),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())
  if load:
    topic_model_pr.load_weights(load)
  #topic_model_pr.layers[-3].set_weights([np.zeros((2048,1000))])
  #topic_model_pr.layers[-3].trainable = False
  return topic_model_pr, optimizer_reset, optimizer, topic_vector_n,  n_concept, f_input

def topic_model_new(predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  # f_input size (None, 8,8,2048)
  #input = Input(shape=(299,299,3), name='input')
  #f_input = get_feature(input)

  f_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(f_input)
  # topic vector size (2048,n_concept)
  topic_vector = Weight((f_train.shape[3], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)

  # topic prob = batchsize * 8 * 8 * n_concept
  #topic_prob = Weight_instance((n_concept))(f_input)
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  #topic_prob_pos = Lambda(lambda x: K.maximum(x,-1000))(topic_prob)
  #print(K.sum(topic_prob, axis=3, keepdims=True))
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=3, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  # rec size is batchsize * 8 * 8 * 2048
  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[3]))(f_input)
  #rec = Lambda(lambda x:K.dot(x[0],K.transpose(x[1])))([topic_prob_pos, topic_vector])
  #scale_value = Weight((1,1,1,1))(f_input)
  #bias_value = Weight((1,1,1,2048))(f_input)
  #scaled_rec1 = Lambda(lambda x: x[0] * x[1])([rec, scale_value])
  #scaled_rec2 = Lambda(lambda x: x[0] + x[1])([scaled_rec1, bias_value])
  rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  #rec_layer_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(rec_layer)
  pred = predict(rec_layer_2)
  topic_model_pr = Model(inputs=f_input, outputs=pred)
  topic_model_pr.layers[-1].trainable = True
  #topic_model_pr.layers[1].trainable = False
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    # These depend on the optimizer class
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  # Later when you want to reset the optimizer
  #K.get_session().run(optimizer_reset)
  #print(metric1)
  metric1.append(mean_sim(topic_prob_n, n_concept))
  topic_model_pr.compile(
      loss=topic_loss(topic_prob_n, topic_vector_n,  n_concept, f_input, loss1=loss1),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())
  if load:
    topic_model_pr.load_weights(load)
  #topic_model_pr.layers[-3].set_weights([np.zeros((2048,1000))])
  #topic_model_pr.layers[-3].trainable = False
  return topic_model_pr, optimizer_reset, optimizer, topic_vector_n,  n_concept, f_input

def topic_model_nlp(predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  # f_input size (None, 8,8,2048)
  #input = Input(shape=(299,299,3), name='input')
  #f_input = get_feature(input)

  f_input = Input(shape=(f_train.shape[1],f_train.shape[2]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(2)))(f_input)
  # topic vector size (2048,n_concept)
  topic_vector = Weight((f_train.shape[2], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)

  # topic prob = batchsize * 8 * 8 * n_concept
  #topic_prob = Weight_instance((n_concept))(f_input)
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  #topic_prob_pos = Lambda(lambda x: K.maximum(x,-1000))(topic_prob)
  #print(K.sum(topic_prob, axis=3, keepdims=True))
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=2, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  # rec size is batchsize * 8 * 8 * 2048
  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[2]))(f_input)
  rec_layer_1 = Lambda(lambda x:(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  rec_layer_f2 = Flatten()(rec_layer_2)
  #rec_layer_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(rec_layer)
  pred = predict(rec_layer_f2)
  topic_model_pr = Model(inputs=f_input, outputs=pred)
  topic_model_pr.layers[-1].trainable = True
  #topic_model_pr.layers[1].trainable = False
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    # These depend on the optimizer class
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  # Later when you want to reset the optimizer
  #K.get_session().run(optimizer_reset)
  #print(metric1)
  metric1.append(mean_sim(topic_prob_n, n_concept))
  topic_model_pr.compile(
      loss=topic_loss_nlp(topic_prob_n, topic_vector_n,  n_concept, f_input, loss1=loss1),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())
  if load:
    topic_model_pr.load_weights(load)
  #topic_model_pr.layers[-3].set_weights([np.zeros((2048,1000))])
  #topic_model_pr.layers[-3].trainable = False
  return topic_model_pr, optimizer_reset, optimizer, topic_vector_n,  n_concept, f_input

def topic_model_shap(predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  last_dim = len(f_train.shape)-1
  print(last_dim)
  f_input = Input(shape=(f_train.shape[1:]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(last_dim)))(f_input)
  # topic vector size (2048,n_concept)
  topic_vector = Weight((f_train.shape[-1], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)

  # topic prob = batchsize * 8 * 8 * n_concept
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  #topic_prob_pos = Lambda(lambda x: K.maximum(x,-1000))(topic_prob)
  #print(K.sum(topic_prob, axis=3, keepdims=True))
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=last_dim, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  # rec size is batchsize * 8 * 8 * 2048
  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[last_dim]))(f_input)
  rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  if last_dim==2:
    rec_layer_2 = Flatten()(rec_layer_2)
  #rec_layer_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(rec_layer)
  pred = predict(rec_layer_2)
  topic_model_pr = Model(inputs=f_input, outputs=pred)
  topic_model_pr.layers[-1].trainable = False
  topic_model_pr.layers[1].trainable = False
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    # These depend on the optimizer class
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  # Later when you want to reset the optimizer
  #K.get_session().run(optimizer_reset)
  #print(metric1)
  metric1.append(mean_sim(topic_prob_n, n_concept))
  topic_model_pr.compile(
      loss=given_loss( loss1=loss1),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())
  if load:
    topic_model_pr.load_weights(load)
  #topic_model_pr.layers[-3].trainable = False
  return topic_model_pr

def topic_model_new_toy(predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               metric1=['accuracy'],
               opt='adam',
               loss1=tf.nn.softmax_cross_entropy_with_logits,
               thres=0.0,
               load=False,
               para = 0.5):
  """Returns main function of topic model."""


  f_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input')
  f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(f_input)

  topic_vector = Weight((f_train.shape[3], n_concept))(f_input)
  topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)
  topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  topic_prob_sum = Lambda(lambda x: K.sum(x, axis=3, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])

  rec_vector_1 = Weight((n_concept, 500))(f_input)
  rec_vector_2 = Weight((500, f_train.shape[3]))(f_input)
  rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  pred = predict(rec_layer_2)
  topic_model_pr = Model(inputs=f_input, outputs=pred)
  topic_model_pr.layers[-1].trainable = False
  if opt =='sgd':
    optimizer = SGD(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr,
          optimizer.momentum, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  elif opt =='adam':
    optimizer = Adam(lr=0.001)
    optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                             optimizer.beta_2, optimizer.decay]
    optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  metric1.append(mean_sim(topic_prob_n, n_concept))
  topic_model_pr.compile(
      loss=topic_loss_toy(topic_prob_n, topic_vector_n,  n_concept, f_input, loss1=loss1, para = para),
      optimizer=optimizer,metrics=metric1)
  print(topic_model_pr.summary())
  if load:
    topic_model_pr.load_weights(load)
  return topic_model_pr, optimizer_reset, optimizer, topic_vector_n,  n_concept, f_input

def get_acc(binary_sample, f_val, y_val_logit, shap_model, verbose=False):
  """Returns accuracy."""
  acc = shap_model.evaluate(
      [f_val, np.tile(np.array(binary_sample), (f_val.shape[0], 1))],
      y_val_logit,
      verbose=verbose)[1]
  print(acc)
  return acc

def shap_kernel(n, k):
  """Returns kernel of shapley in KernelSHAP."""
  return (n-1)*1.0/((n-k)*k*comb(n, k))

def shap_kernel_adjust(n, k, p=0.5):
  """Returns kernel of shapley in KernelSHAP."""
  return (n-1)*1.0/((n-k)*k*comb(n, k)) / (np.power(p,k)*np.power(1-p,n-k))

def get_shap(nc, f_train, y_train, f_val, y_val, topic_vec, model_shap, full_acc, null_acc, n_concept, get_acc_f):
  """Returns ConceptSHAP."""
  inputs = list(itertools.product([0, 1], repeat=n_concept))
  #\binary_sample, topic_vec, f_train, y_train, f_val, y_val, model_shap, verbose=False)
  outputs = [(get_acc_f(k, topic_vec, f_train, y_train, f_val, y_val, model_shap, verbose=False)-null_acc)/
             (full_acc-null_acc) for k in inputs]
  kernel = [shap_kernel(nc, np.sum(ii)) for ii in inputs]
  x = np.array(inputs)
  y = np.array(outputs)
  k = np.array(kernel)
  k[k == inf] = 10000
  xkx = np.matmul(np.matmul(x.transpose(), np.diag(k)), x)
  xky = np.matmul(np.matmul(x.transpose(), np.diag(k)), y)
  expl = np.matmul(np.linalg.pinv(xkx), xky)
  return expl


def main(_):
  return


if __name__ == '__main__':
  app.run(main)
