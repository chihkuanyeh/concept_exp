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

# lint as: python3
"""Main file to run AwA experiments."""

import copy
import ipca_v2
import imdb_helper_v2
import keras
#import keras.backend as K
import itertools
import os

import numpy as np
from absl import app
import matplotlib.pyplot as plt
import parse_imdb

stop_word_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", 
                  "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
                  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", 
                  "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
                  "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
                  "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",  
                  "only", "own", "same", "so", "than", "very", "s", "t", "can", "will", "just", "don", "should", "now",


                  "one", "it's", "br", "<PAD>", "<START>", "<UNK>", "would", "could", "also", "may", "many", "go", "another",
                  "want", "two", "actually", "every", "thing", "know", "made", "get", "something", "back", "though"]


def main(_):
  n_concept = 4
  batch_size = 128
  pretrain = True
  thres = 0.3
  x_train, x_val, y_train, y_val = imdb_helper_v2.load_data()
  print(x_train[0])

  # Loads model
  model, feature_model, predict_model = imdb_helper_v2.load_model_stm(
        x_train, y_train, x_val, y_val, pretrain=pretrain)

  pretrain = False
  # get feature
  if not pretrain:
    f_train = feature_model.predict(x_train)
    f_val = feature_model.predict(x_val)
    np.save('f_train_imdb.npy', f_train)
    np.save('f_val_imdb.npy', f_val)
    #np.save('all_feature_best.npy', all_feature)
  else:
    f_train = np.load('f_train_imdb.npy')
    f_val = np.load('f_val_imdb.npy')

  N = f_train.shape[0]
  f_train = f_train.reshape(-1,196,250)
  f_val = f_val.reshape(-1,196,250)
  print(f_train.shape)
  trained = True
  thres_array = [0.3]
    
  if not trained:
    for count,thres in enumerate(thres_array):
      if count:
        load = 'latest_topic_nlp.h5'
      else:
        load = False
      #load = 'latest_topic_nlp.h5'
      topic_model_pr, optimizer_reset, optimizer, \
          topic_vector,  n_concept, f_input = ipca_v2.topic_model_nlp(predict_model,
                                        f_train,
                                        y_train,
                                        f_val,
                                        y_val,
                                        n_concept,
                                        verbose=False,
                                        epochs=10,
                                        metric1=['accuracy'],
                                        loss1=keras.losses.binary_crossentropy,
                                        thres=thres,
                                        load=load)

      topic_model_pr.fit(
          f_train,
          y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(f_val, y_val),
          verbose=True)
      #K.get_session().run(optimizer_reset)
      
      topic_model_pr.save_weights('latest_topic_nlp.h5')

    topic_vec = topic_model_pr.layers[1].get_weights()[0]
    recov_vec = topic_model_pr.layers[-4].get_weights()[0]
    np.save('topic_vec_nlp.npy',topic_vec)
    np.save('recov_vec_nlp.npy',recov_vec)
  else:
    topic_vec = np.load('topic_vec_nlp.npy')
    recov_vec = np.load('recov_vec_nlp.npy')


  model_shap = ipca_v2.topic_model_shap(predict_model,
                                        f_train,
                                        y_train,
                                        f_val,
                                        y_val,
                                        n_concept,
                                        verbose=False,
                                        epochs=0,
                                        metric1=['accuracy'],
                                        loss1=keras.losses.binary_crossentropy,
                                        thres=thres,
                                        load='latest_topic_nlp.h5') 
  topic_vec = np.load('topic_vec_nlp.npy')
  w_3 = model_shap.layers[-3].get_weights()
  w_5 = model_shap.layers[-5].get_weights()  

  n_sample = 16 # 2^n_concept
  trained_shap = True
  classes=2
  predictions = []
  def get_acc_nlp(binary_sample, topic_vec, f_train, y_train, f_val, y_val, model_shap, verbose=False):
    topic_vec_temp = copy.copy(topic_vec) 
    topic_vec_temp[:,np.array(binary_sample)==0] = 0
    model_shap.layers[1].set_weights([topic_vec_temp])
    his = model_shap.fit(
          f_train,
          y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=verbose)
    prediction = model_shap.predict(f_val)
    acc = np.sum((prediction[:,0]>0.5)==y_val)*1.0/12500
    return acc

  expl = ipca_v2.get_shap(n_concept, f_train, y_train, f_val, y_val, topic_vec, model_shap, 0.902, 0.5, n_concept, get_acc_nlp)
  print(expl)
  
       
if __name__ == '__main__':
  app.run(main)