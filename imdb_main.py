# lint as: python3
"""Main file to run AwA experiments."""
import os
import ipca_v2
import imdb_helper_v2
import keras
import keras.backend as K

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
  
  # Loads data.
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
    np.save('imdb_data/f_train_imdb.npy', f_train)
    np.save('imdb_data/f_val_imdb.npy', f_val)
    #np.save('all_feature_best.npy', all_feature)
  else:
    f_train = np.load('imdb_data/f_train_imdb.npy')
    f_val = np.load('imdb_data/f_val_imdb.npy')

  N = f_train.shape[0]
  f_train = f_train.reshape(-1,196,250)
  f_val = f_val.reshape(-1,196,250)
  print(f_train.shape)
  trained = True
  thres_array = [0.3]
    
  if not trained:
    for count,thres in enumerate(thres_array):
      if count:
        load = 'imdb_data/latest_topic_nlp.h5'
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
                                        metric1=['binary_accuracy'],
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
      
      topic_model_pr.save_weights('imdb_data/latest_topic_nlp.h5')

    topic_vec = topic_model_pr.layers[1].get_weights()[0]
    recov_vec = topic_model_pr.layers[-4].get_weights()[0]
    np.save('imdb_data/topic_vec_nlp.npy',topic_vec)
    np.save('imdb_data/recov_vec_nlp.npy',recov_vec)
  else:
    topic_vec = np.load('imdb_data/topic_vec_nlp.npy')
    recov_vec = np.load('imdb_data/recov_vec_nlp.npy')

  
  f_train_n = f_train/(np.linalg.norm(f_train,axis=2,keepdims=True)+1e-9)
  topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
  topic_prob = np.matmul(f_train_n,topic_vec_n)
  print(topic_prob.shape)
  print('top prob')
  print(np.mean(np.max(topic_prob,axis=(0,1))))
  n_size = 196
  concept_nn_array = np.zeros((n_concept,500,9))
  for i in range(n_concept):
    print('concept:{}'.format(i))
    image_list = []
    ind = np.argsort(topic_prob[:,:,i].flatten())[::-1][:500]
    #ind = np.argpartition(topic_prob[:,:,i].flatten(), -10)[-10:]
    sim_list = topic_prob[:,:,i].flatten()[ind]
    print(sim_list)
    print(ind)
    dict_count = {}
    for jc,j in enumerate(ind):
        j_int = int(np.floor(j/(n_size)))
        a = int(j-j_int*(n_size))
        temp_sentence = imdb_helper_v2.show_sentence(x_train, j_int, a, dict_count)
        concept_nn_array[i,jc,:] = x_train[j_int][a*2:a*2+9]
        #f1 = imagedir+filename[j_int]
    for key in dict_count:
      if dict_count[key]>=8 and key not in stop_word_list:
        print(key)
        print(dict_count[key])
  np.save('concept_nn_nlp.npy', concept_nn_array)
       
if __name__ == '__main__':
  app.run(main)