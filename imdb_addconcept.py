# lint as: python3
"""Main file to run AwA experiments."""
import os
import copy
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

  # load nn from concepts
  concept_nn_array = np.load('concept_nn_nlp.npy')

  
  p_val = model.predict(x_val)
  print(np.mean(p_val))
  print(np.mean(p_val>0.5))

  for concept in range(n_concept):
    x_val_temp = copy.copy(x_val)
    x_val_temp[:,:-45] = x_val[:,45:]
    x_val_temp[:,-45:-36] = concept_nn_array[concept,np.random.choice(500, x_val.shape[0]),:]    
    x_val_temp[:,-36:-27] = concept_nn_array[concept,np.random.choice(500, x_val.shape[0]),:]    
    x_val_temp[:,-27:-18] = concept_nn_array[concept,np.random.choice(500, x_val.shape[0]),:]
    x_val_temp[:,-18:-9] = concept_nn_array[concept,np.random.choice(500, x_val.shape[0]),:]
    x_val_temp[:,-9:] = concept_nn_array[concept,np.random.choice(500, x_val.shape[0]),:]
    p_val = model.predict(x_val_temp)
    print(np.mean(p_val))    
    print(np.mean(p_val>0.5))
if __name__ == '__main__':
  app.run(main)