
## lint as: python3
"""Main file to run AwA experiments."""
import copy
import os
import awa_helper_v2
import ipca_v2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from matplotlib import gridspec

DATA_DIR = '/volume00/jason/Animals_with_Attributes2/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
VALID_DIR = os.path.join(DATA_DIR, 'val/')
SIZE = (299, 299)
batch_size = 64
n_concept = 70
pretrained = True
trainable = True

base_folder = '/volume00/jason/conceptshap_release/'

if __name__ == '__main__':

  feature_model, predict_model, finetuned_model = awa_helper_v2.load_model_stm_small(-148, trainable)

  y_train_logit, y_val_logit, y_train, \
      y_val, f_train, _, f_val, batches, batches_fix_val = awa_helper_v2.load_data(TRAIN_DIR,
                                                   SIZE,
                                                   batch_size,
                                                   feature_model,
                                                   predict_model,
                                                   finetuned_model,
                                                   pretrained,
                                                   noise=0.0)

  N = f_train.shape[0]
  N_val = f_val.shape[0]
  #print(f_train_crop.shape)
  thres_array = [0.5]

  #f_train_crop = np.reshape(f_train_crop, (-1,4,8,8,2048))


  for count,thres in enumerate(thres_array):
    # set load = False to train from scratch
    load = 'awa_data/latest_topic_awa_nocrop_small_70_new2.h5'
    topic_model_pr, optimizer_reset, optimizer, \
        topic_vector,  n_concept, f_input = ipca_v2.topic_model_new(predict_model,
                                      f_train,
                                      y_train_logit[:N],
                                      f_val,
                                      y_val_logit[:f_val.shape[0]],
                                      n_concept,
                                      verbose=False,
                                      epochs=30,
                                      metric1=['accuracy'],
                                      loss1=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2,
                                      thres=thres,
                                      load=load)

  topic_vec = topic_model_pr.layers[1].get_weights()[0]
  np.save('awa_data/topic_vec_awa_small_70_new2.npy',topic_vec)
  #directly load topic_vec
  #topic_vec = np.load('awa_data/topic_vec_awa_small_70_new2.npy')

  topic_vec_n = topic_vec/np.linalg.norm(topic_vec,axis=0,keepdims=True)
  aa = np.matmul(topic_vec_n.T,topic_vec_n)- np.eye(70)
  # remove similar concepts
  remove_list = set()
  for i in range(n_concept):
    for j in range(i+1,n_concept):
      if aa[i,j]>0.95:
        remove_list.add(j)
  print('remove_list',remove_list)

  all = set(range(70))
  keep = list(all -remove_list)
  rm = list(remove_list)
  print('keep list', keep)
  n_concept_alive = len(keep)
  # remove repeated concepts
  topic_vec_n = topic_vec_n[:,keep]
  keep_array = np.array(keep)


  # visualize awa concepts
  #f_train_crop = f_train_crop[:5000]
  f_train_n = f_train/np.linalg.norm(f_train,axis=3,keepdims=True)
  topic_prob = np.matmul(f_train_n,topic_vec_n)
  print('top prob')
  print(np.mean(np.max(topic_prob,axis=(0,1,2))))

  # get shapley value for each concept
  # create a new model to calculate completeness for a set of concepts efficiently

  model_shap = ipca_v2.topic_model_shap(predict_model,
                                        f_train,
                                        y_train_logit[:N],
                                        f_val,
                                        y_val_logit[:f_val.shape[0]],
                                        n_concept,
                                        verbose=False,
                                        epochs=0,
                                        metric1=['accuracy'],
                                        loss1=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2,
                                        thres=0,
                                        load='awa_data/latest_topic_awa_nocrop_small_70_new2.h5')
  topic_vec = np.load('awa_data/topic_vec_awa_small_70_new2.npy')
  w_3 = model_shap.layers[-3].get_weights()
  w_5 = model_shap.layers[-5].get_weights()

  # sample of points to estimate Shapley value
  n_sample = 100
  classes = 50
  trained_shap = True
  predictions = []
  output = np.zeros((n_sample,classes))
  if not trained_shap:
    binary_input = ipca_v2.sample_binary(n_concept_alive, n_sample, 0.2)
    binary_input[-1,:] = 1
    for i in range(n_sample):
      topic_vec_temp = copy.copy(topic_vec)
      topic_vec_temp[:,rm] = 0
      topic_vec_temp[:,keep_array[binary_input[i,:]==0]] = 0
      model_shap.layers[1].set_weights([topic_vec_temp])
      model_shap.layers[-3].set_weights(w_3)
      model_shap.layers[-5].set_weights(w_5)
      his = model_shap.fit(
          f_train,
          y_train_logit[:N],
          batch_size=batch_size,
          epochs=6,
          verbose=True)
      prediction = model_shap.predict(f_val)
      predictions.append(prediction)
      print(prediction.shape)
      #print(his.history['val_accuracy'])
      #output.append(his.history['val_accuracy'][-1])
    #np.save('output_list_0_2',output)
    np.save('awa_data/predictions_0_2_100.npy',predictions)
    np.save('awa_data/binary_input_0_2_100.npy', binary_input)
  else:
    predictions = np.load('awa_data/predictions_0_2_100.npy')
    binary_input = np.load('awa_data/binary_input_0_2_100.npy')
    predictions = predictions[:,:N_val,:]
    #print(predictions[0].shape)
  kernel = [ipca_v2.shap_kernel_adjust(n_concept_alive, ii, 0.2) for ii in np.sum(binary_input,axis=1)]
  classes_index = []
  for classid in range(50):
    classes_index.append(y_val_logit[:N_val,classid]==1)
    print((y_val_logit[:N_val,classid]==1).shape)
  for count, prediction in enumerate(predictions):
    for classid in range(50):
      output[count,classid] = (np.sum(np.argmax(prediction[classes_index[classid]], axis=1)==classid)*1.0/N_val)

  print('output', output)

  x = np.array(binary_input)
  print('kernel', kernel)
  expl_array = np.zeros((50,n_concept_alive))
  for classid in range(50):
    print(classid)
    y = np.array(output[:,classid])
    k = np.array(kernel)
    k[k == np.inf] = 100000
    xkx = np.matmul(np.matmul(x.transpose(), np.diag(k)), x)
    xky = np.matmul(np.matmul(x.transpose(), np.diag(k)), y)
    expl_array[classid,:] = np.matmul(np.linalg.pinv(xkx), xky)
    print(np.sum(expl_array[classid]))
  np.save('awa_data/expl_array.npy', expl_array)
  classes = np.load('awa_data/classes.pickle', allow_pickle=True)
  work_dir = 'work_awa_70'
  dir_exist = {}
  for classid in range(50):
    work_dir_class = work_dir+'_'+classes[classid]
    try:
      # Create target Directory
      os.mkdir('test_shap_2/'+work_dir_class)
      print("Directory " , work_dir_class,  " Created ")
    except FileExistsError:
      print("Directory " , work_dir_class ,  " already exists")
    class_start = np.min(np.where(y_train_logit[:,classid]==1))
    filename = np.load('awa_data/file_name.npy')
    imagedir = '/volume00/jason/Animals_with_Attributes2/train/'
    top_concept_list = (-expl_array[classid,:]).argsort()[:20]
    plot_concept_list = []
    concept_array = []
    concept_score = []
    concept_count = 0
    for i in top_concept_list:
      ind = (-topic_prob[y_train_logit[:,classid]==1,:,:,i].flatten()).argsort()[:500]
      shapscore = expl_array[classid,i]
      print('shapscore', shapscore)
      print('for concept {}'.format(i))
      #print(topic_prob[y_train[:N]==classid,:,:,i].flatten()[ind])
      image_list = []
      plot_count = 0
      plot_list = []
      score_count = []
      for jc,jj in enumerate(ind):
          j = jj+class_start*17*17
          j_int = int(np.floor(j/(17*17)))
          if j_int not in plot_list:
            #print(topic_prob[y_train[:N]==classid,:,:,i].flatten()[jj])
            score_count.append(topic_prob[y_train_logit[:,classid]==1,:,:,i].flatten()[jj])
            #print(score_count[-1])
            plot_list.append(j_int)
            plot_count+=1
          else:
            continue
          rem = j-j_int*17*17
          dim = np.floor(rem/289)
          a = int((rem-dim*289)/17)
          b = int((rem-dim*289)%17)
          if dim ==0:
            da = 0
            db = 0
          if dim ==1:
            da = 0
            db = 1
          if dim ==2:
            da = 1
            db = 0
          else:
            da = 1
            db = 1
          f1 = imagedir+filename[j_int]
          f2 = base_folder + 'test_shap_2/{}/concept_{}_{}.png'.format(work_dir_class,i,jc)
          path = base_folder + 'test_shap_2/{}'.format(work_dir_class)
          if path not in dir_exist:
            try:
              os.mkdir(path)
              dir_exist[path] = 1
            except OSError as error:
              print(error)
          awa_helper_v2.copy_save_image(f1,f2,a,b,0,0)
          image_list.append(plt.imread(f2))
          if plot_count ==5:
            break
      #fig, axs = plt.subplots(1,5, figsize=(30, 10))
      nrow = 1
      ncol = 5

      fig = plt.figure(figsize=(ncol +1, nrow+1))

      gs = gridspec.GridSpec(nrow, ncol,
              wspace=0.05, hspace=0.05,
              top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
              left=0.5/(ncol+1), right=1-0.5/(ncol+1))
      for ii in range(5):
        ax= plt.subplot(gs[0,ii])
        ax.axis('off')
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.imshow(image_list[ii])
      f3 = base_folder + 'test_shap_2/{}/concept_{}_all.png'.format(work_dir_class,i)
      print(i)
      plt.axis('off')
      plt.savefig(f3)
      print(score_count)
      print(concept_count)
      if np.mean(np.array(score_count))>0.8 and concept_count <=2:
        concept_count += 1
        plot_concept_list.append(image_list)
        concept_array.append(i)
        concept_score.append(shapscore)

    nrow = 3
    ncol = 5
    fig = plt.figure(figsize=(ncol+1, nrow+1+1))
    #fig.subplots_adjust(top = 2)
    fig.suptitle('{}'.format(classes[classid]), fontsize = 25, y=1.08)
    plt.rcParams.update({'font.size': 20})
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.05, hspace=0.32)
    for ii in range(concept_count):
      for jj in range(5):
        ax= plt.subplot(gs[ii,jj])
        if jj ==2:
          ax.set_title('Concept {0}    {1:.4f}'.format(concept_array[ii], concept_score[ii]), loc='center')
        ax.axis('off')
        ax.imshow(plot_concept_list[ii][jj])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    f4 = base_folder + 'test_shap_2/concept_all_{}.png'.format(work_dir_class)
    print(i)
    plt.axis('off')
    plt.savefig(f4)
