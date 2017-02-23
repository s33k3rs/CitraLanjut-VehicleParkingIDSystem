from __future__ import print_function

import Inception_PCL_FlowXY as incp

import numpy as np
import time
import warnings
import csv
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import os
import fnmatch
from skimage import io
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_similarity_score
from sklearn.utils import compute_class_weight
import itertools

from keras.models import Model, Sequential
from keras.layers import Flatten, GlobalAveragePooling2D, Dropout, Dense, Input, BatchNormalization, LeakyReLU, merge
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D, AveragePooling2D
#from keras.preprocessing import image
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

Base_Path = '/home/ihsan/Documents/MIK/PCL_Project/'
Dataset_Path = Base_Path + 'Data/'
Base_DREAM_Path = '/home/ihsan/Documents/MIK/DREAM/'
Training_Set_Path = Dataset_Path + 'Train'
Test_Set_Path = Dataset_Path + 'Test'
BATCH_SIZE=42

'''this is basically the same as the Inception_PCL_FlowXY but without the training part.
This loads the best weights for each method (the one that resulted in the minimum validation loss),
then goes and makes the confusion matrices for both training and test sets.

The original purpose was to speed up the troubleshooting for the confusion matrices
(because it's a lot less accurate than the Keras prediction scores lead you to believe...
'''

if __name__ == '__main__':
    img_width=299
    img_height=299

    pp_method_list= ['_PPd_RGB', '_PPd_G', '_PPd_G_Canny', '_PPd_Sobel_Each','_PPd_Sobel_HSV']
    #pp_method_list = ['_PPd_Sobel_HSV']

    class_list = ['0_NoCar','11_Jazz','12_CRV','13_Civic','1_NotinDB','20_Livina','21_Juke','22_XTrail','2_Ayla','3_Avanza','4_Kijang',
     '5_Yaris','6_Corolla','9_Brio'] #CUSTOM BECAUSE VALIDATION GENERATOR SORTS THE FOLDERS ALPHABETICALLY

    for ppmethod in pp_method_list:
        train_data_dir = Dataset_Path + 'Train' + ppmethod + '/'
        test_data_dir = Dataset_Path + 'Test' + ppmethod + '/'
        print('{} and {}'.format(train_data_dir,test_data_dir))

        Best_Model_Weights_Path= Base_Path + '/Weights/PCL_IpV3' + ppmethod +'.h5'

        train_datagen=ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,channel_shift_range=5.0, rotation_range=10)
        test_datagen=ImageDataGenerator(rescale=1./255,horizontal_flip=True,channel_shift_range=0.5)

        print("initializing validation generator")
        validation_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height),
            batch_size=42, class_mode='categorical', shuffle=True,seed=1337)
        nb_testing_samples = validation_generator.nb_sample

        print("initializing training generator")
        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
            batch_size=42, class_mode='categorical',shuffle=True,seed=1337)
        nb_training_samples = train_generator.nb_sample

        full_model = incp.BuildWithConvClassifier(Best_Model_Weights_Path, load_weights=False)

        full_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        # full_model.summary()

        for layer in full_model.layers:
            layer.trainable = False  # freeze the entire inception model for now.

        #217 if you just want to train the top convnet #172 for mixed8 #158 for mixed 7
        for layer in full_model.layers[217:233]: #add +1 to the last layer index.. otherwise the sigmoid won't get trained!
            layer.trainable = True  # Train only the classifier
        #
        # for i, layer in enumerate(full_model.layers):
        #     print(i, layer.name, "Trainable?: ", layer.trainable)

        class_weight_to_fit = incp.GetDataRatio(ppmethod, class_list, nb_training_samples, 'Train')
        #print('class_weight_to_fit: {}'.format(class_weight_to_fit))


        #Now we test the best weight. "Instrumented best weight run"
        full_model.load_weights(Best_Model_Weights_Path) #LOAD (not set) the best weights.
        X_test, Y_test = incp.GetXYData(ppmethod,nb_testing_samples,class_list,'Test') #get them big arrays
        y_predictions = full_model.predict(X_test, batch_size=BATCH_SIZE)
        Y_test_categorical = np_utils.to_categorical(Y_test,len(class_list))
        #print('Y_test_categorical:{}'.for)
        #print('y_predictions before conversion: {}'.format(y_predictions))
        y_preds_as_classes = np_utils.categorical_probas_to_classes(y_predictions) #converts class probs to one-hot
        print('y_predictions after conversion: {}'.format(y_preds_as_classes))
        #print('pre-converted preds: {}'.format(y_predictions))
        print ('converted preds: {}'.format(y_preds_as_classes))
        print ('ground truth labels: {}'.format (Y_test))


        test_comparo = np.zeros((len(y_preds_as_classes), 2), dtype=np.uint8)
        for i in range(0, len(y_preds_as_classes) - 1):  # first column is truth. 2nd is predict.
            test_comparo[i, 0] = Y_test[i]
            test_comparo[i, 1] = y_preds_as_classes[i]
        np.savetxt(Base_Path + 'Results/TestSetPredsComparo' + ppmethod + '.txt', test_comparo,
                   fmt='%1i', delimiter=' ', newline='\n', header='truth, pred', footer=ppmethod, comments='# ')

        conf_matrix_testset = confusion_matrix(Y_test, y_preds_as_classes)
        np.savetxt(Base_Path + 'Results/CfMat_Test' + ppmethod + '.txt', conf_matrix_testset,
                   fmt='%1i', delimiter=' ', newline='\n', header='', footer='', comments='# ')
        plt.figure()
        incp.plot_confusion_matrix(conf_matrix_testset, class_list, normalize=False,
                              title='Confusion Matrix Test' + str(ppmethod), cmap=plt.cm.Blues)
        plt.clf()

        test_score= accuracy_score(Y_test, y_preds_as_classes, normalize=True, sample_weight=None)
        test_jaccard=jaccard_similarity_score(Y_test, y_preds_as_classes,normalize=True)
        print('test score: {} test Jacccard: {}'.format(test_score, test_jaccard))


        #Now get the confusion matrix for the training set batch by batch..
        X_train_full, Y_train_full = incp.GetXYData(ppmethod, nb_training_samples, class_list,'Train')
        # print('xtrainfull shape: {} ndim:{} ytrainfullshape: {} ndim {}'.format(X_train_full.shape, X_train_full.ndim,
        #                                                                         Y_train_full.shape, Y_train_full.ndim))
        #huge array and vector.. need to break into chunks.
        #there are 1738 training images. Training batch size is 21 (since test set is 42 large) -> only 1722 images used for this.

        train_conf_mat_batch_size = BATCH_SIZE
        X_train_chunk = np.zeros((train_conf_mat_batch_size,3,img_height,img_width),dtype=np.uint8)
        Y_train_chunk = np.zeros(train_conf_mat_batch_size)
        batch_counter = train_conf_mat_batch_size
        y_preds_initialized = False
        Y_train_cut = Y_train_full[0:-16] #0:-16 WITH THE REAL SET!! #1738 in total, deduct 16 values fron the right -> 1722.

        for i in range(0,1722-1): #len(Y_train_full) 1722-1
            if batch_counter > 0:
                X_train_chunk[train_conf_mat_batch_size - batch_counter, 0, :, :] = X_train_full[i, 0, :, :]
                X_train_chunk[train_conf_mat_batch_size - batch_counter, 1, :, :] = X_train_full[i, 1, :, :]
                X_train_chunk[train_conf_mat_batch_size - batch_counter, 2, :, :] = X_train_full[i, 2, :, :]
                Y_train_chunk[train_conf_mat_batch_size - batch_counter] = Y_train_full[i]
                batch_counter -= 1
                #print('first if i: {} batch_counter: {}'.format(i,batch_counter))
            if batch_counter == 0:
                y_preds_chunk = full_model.predict(X_train_chunk, batch_size=train_conf_mat_batch_size)
                y_preds_chunk_converted = np_utils.categorical_probas_to_classes(y_preds_chunk)
                batch_counter = train_conf_mat_batch_size
                #print('second if i: {} batch_counter: {}'.format(i, batch_counter))
                if y_preds_initialized == False: #first batch to be predicted
                    y_preds_full = y_preds_chunk_converted
                    y_preds_initialized=True
                    print('y_preds_full:{}'.format(y_preds_full))
                    #print('y_preds_initialized: {}'.format(y_preds_initialized))
                if y_preds_initialized==True:
                    y_preds_full=np.append(y_preds_full,y_preds_chunk_converted)
                    #print('y_preds_full:{}'.format(y_preds_full))
                    #print('appended')

        print ('converted preds: {}'.format(y_preds_full))
        print ('ground truth labels: {}'.format (Y_train_cut))
        
        train_comparo = np.zeros((len(y_preds_full),2), dtype=np.uint8)
        for i in range(0, len(y_preds_full) - 1): #first column is truth. 2nd is predict. 
            train_comparo[i,0]=Y_train_cut[i]
            train_comparo[i,1]=y_preds_full[i]
        
        np.savetxt(Base_Path + 'Results/TrainSetPredsComparo' + ppmethod + '.txt', train_comparo,
                   fmt='%1i', delimiter=' ', newline='\n', header='truth, pred', footer=ppmethod, comments='# ')
            

        conf_matrix_trainset = confusion_matrix(Y_train_cut, y_preds_full)
        incp.plot_confusion_matrix(conf_matrix_trainset, class_list, normalize=False,
                              title='Confusion Matrix Train' + str(ppmethod), cmap=plt.cm.PuRd)
        plt.clf()
        train_score= accuracy_score(Y_train_cut, y_preds_full, normalize=True, sample_weight=None)
        train_jaccard=jaccard_similarity_score(Y_train_cut, y_preds_full,normalize=True)
        print('train score: {} train Jacccard: {}'.format(train_score, train_jaccard))


