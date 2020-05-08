# lint as: python3
"""Main file to run AwA experiments."""
import os
import toy_helper_v2
import ipca_v2
import keras
import keras.backend as K

import numpy as np
from sklearn.decomposition import PCA
#from fbpca import diffsnorm, pca
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from absl import app

def main(_):
  n_concept = 5
  n_cluster = 5
  n = 60000
  n0 = int(n * 0.8)
  batch_size = 128
  pretrain = True
  verbose = True
  thres = 0.2
  # create dataset
  #toy_helper_v2.create_dataset(n_sample=60000)
  
  # Loads data.
  x, y, concept = toy_helper_v2.load_xyconcept(n, pretrain)
  if not pretrain:
    x_train = x[:n0, :]
    x_val = x[n0:, :]
  y_train = y[:n0, :]
  y_val = y[n0:, :]

  # Loads model
  if not pretrain:
    feature_model, predict_model = toy_helper_v2.load_model_stm_new(
        x_train, y_train, x_val, y_val, pretrain=pretrain)
  else:
    feature_model, predict_model = toy_helper_v2.load_model_stm_new(_, _, _, _, pretrain=pretrain)

  # get feature
  if not pretrain:
    all_feature = feature_model.predict(x)
    np.save('toy_data/all_feature_best.npy', all_feature)
  else:
    all_feature = np.load('toy_data/all_feature_best.npy')
  f_train = all_feature[:n0, :]
  f_val = all_feature[n0:, :]
  print(f_train.shape)
  N = f_train.shape[0]
  trained = False
  para_array = [1.0]
  
  for n_concept in range(5,6,1):
    if not trained:
      for count,para in enumerate(para_array):
        if count:
          load = True
        else:
          load = False

        topic_model_pr, optimizer_reset, optimizer, \
          topic_vector,  n_concept, f_input = ipca_v2.topic_model_new_toy(predict_model,
                                        f_train,
                                        y_train,
                                        f_val,
                                        y_val,
                                        n_concept,
                                        verbose=verbose,
                                        metric1=['binary_accuracy'],
                                        loss1=keras.losses.binary_crossentropy,
                                        thres=thres,
                                        load=False,
                                        para=para)

        topic_model_pr.fit(
          f_train,
          y_train,
          batch_size=batch_size,
          epochs=30,
          validation_data=(f_val, y_val),
          verbose=verbose)

        topic_model_pr.save_weights('toy_data/latest_topic_toy.h5')

        topic_vec = topic_model_pr.layers[1].get_weights()[0]
        recov_vec = topic_model_pr.layers[-3].get_weights()[0]
        topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
        acc = toy_helper_v2.get_groupacc_max(
            topic_vec_n,
            f_train,
            f_val,
            concept,
            n_concept,
            n_cluster,
            n0,
            verbose=verbose)

        ipca_v2.get_completeness(predict_model,
                           f_train,
                           y_train,
                           f_val,
                           y_val,
                           n_concept,
                           topic_vec_n[:,:n_concept],
                           verbose=verbose,
                           epochs=10,
                           metric1=['binary_accuracy'],
                           loss1=keras.losses.binary_crossentropy,
                           thres=thres,
                           load='toy_data/latest_topic_toy.h5')   
    
    # visualize the nearest neighbors
    x = np.load('toy_data/x_data_small.npy')
    f_train_n = f_train[:10000]/(np.linalg.norm(f_train[:10000],axis=3,keepdims=True)+1e-9)
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
    topic_prob = np.matmul(f_train_n,topic_vec_n)
    n_size = 4
    for i in range(n_concept):
      ind = np.argpartition(topic_prob[:,:,:,i].flatten(), -10)[-10:]
      sim_list = topic_prob[:,:,:,i].flatten()[ind]
      for jc,j in enumerate(ind):
        j_int = int(np.floor(j/(n_size*n_size)))
        a = int((j-j_int*(n_size*n_size))/n_size)
        b = int((j-j_int*(n_size*n_size))%n_size)
        f1 = '/volume00/jason/concept_stm/work_toy_test/concept_full_{}_{}.png'.format(i,jc)
        f2 = '/volume00/jason/concept_stm/work_toy_test/concept_{}_{}.png'.format(i,jc)
        #if sim_list[jc]>0.95:
        toy_helper_v2.copy_save_image(x[j_int,:,:,:],f1,f2,a,b)
      np.save('toy_data/topic_vec_toy.npy',topic_vec)
      np.save('toy_data/recov_vec_toy.npy',recov_vec)
    else:
      topic_vec = np.load('toy_data/topic_vec_toy.npy')
      recov_vec = np.load('toy_data/recov_vec_toy.npy')
    
if __name__ == '__main__':
  app.run(main)