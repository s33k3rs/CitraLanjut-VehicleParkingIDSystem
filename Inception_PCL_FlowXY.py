# -*- coding: utf-8 -*-
'''Modified Inception Model from Keras. Theano backend, no top. Initialized with imagenet weights (path is hard coded)
3 options for the top classifier: FC layers, Conv layers, and 3-branch AtrousConv layers.
Build methods will initialize the Inception (if you don't specify a weight to load, it'll load the imagenet weights), then
they'll build the classifier and return the model.
The training call is in the main method (below).

'''
from __future__ import print_function

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
from sklearn.metrics import confusion_matrix
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
#from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import cv2
#from imagenet_utils import decode_predictions

# We use TH_WEIGHTS_PATH_NO_TOP
TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

K.set_image_dim_ordering('th')

Base_Path = '/home/ihsan/Documents/MIK/PCL_Project/'
Dataset_Path = Base_Path + 'Data/'
Base_DREAM_Path = '/home/ihsan/Documents/MIK/DREAM/'
Training_Set_Path = Dataset_Path + 'Train'
Test_Set_Path = Dataset_Path + 'Test'
Processed_Path = Dataset_Path + 'Processed/'
Processed_Training_Set_Path = Dataset_Path + 'Processed/Train'
Processed_Test_Set_Path = Dataset_Path + 'Processed/Test'

# BATCH NORMALIZED 2D CONVOLUTION
def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    '''Utility function to apply conv + BN (batch normalization).
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(mode=2, axis=bn_axis, name=bn_name)(x)
    return x


def InceptionV3(include_top=False, weights='imagenet',
                input_tensor=None):
    '''Instantiate the Inception v3 architecture,
    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 299, 299)  # originally 3, 299, 299
        else:
            input_shape = (3, 299, 299)  # originally 3, None, None.
    else:
        if include_top:
            input_shape = (299, 299, 3)
        else:
            input_shape = (299, 299, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
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

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
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

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = merge([branch3x3_1, branch3x3_2],
                          mode='concat', concat_axis=channel_axis,
                          name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))

    frozen_model = Model(img_input, x)
    #frozen_model.summary()

    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('inception_v3_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='b3baf3070cc4bf476d43a2ea61b0ca5f')
            else:
                weights_path = Base_DREAM_Path + 'Keras_Inception/inception_v3_weights_th_dim_ordering_th_kernels_notop.h5'
                frozen_model.load_weights(weights_path)
                # #'inception_v3_weights_th_dim_ordering_th_kernels_notop.h5',
                #                         TH_WEIGHTS_PATH_NO_TOP,
                #                         cache_subdir='models',
                #                         md5_hash='79aaa90ab4372b4593ba3df64e142f05'

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(frozen_model)
        else:
            if include_top:
                weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='fe114b3ff2ea4bf891e9353d1bbfb32f')
            else:
                weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='2f3609166de1d967d1a481094754f691')
            frozen_model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(frozen_model)

    #print(frozen_model.get_config())
    # print(frozen_model.inputs)
    # print(frozen_model.outputs)

    for layer in frozen_model.layers:
        layer.trainable = False #freeze the entire inception model for now.


    return frozen_model


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + '.png', bbox_inches='tight')

def BuildWithConvClassifier(Conv_Test_Model_Weights_Path, load_weights=False):
    if load_weights == True:
        base_model = InceptionV3(include_top=False, weights=None)
    else:
        print ('using imagenet weights to initialize inception')
        base_model = InceptionV3(include_top=False, weights='imagenet')

    x = base_model.output  # base model is the inceptionV3
    x = Convolution2D(1024,1,1,init='he_normal',activation='relu',border_mode='valid',subsample=(1,1),dim_ordering='th',
                      name='1x1Conv1_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN1')(x)
    x = Convolution2D(1024,2,2,init='he_normal',activation='relu',border_mode='valid',subsample=(2,2),dim_ordering='th',
                      name='Conv2_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN2')(x)
    x = Convolution2D(1024, 2, 2, init='he_normal', activation='relu', border_mode='valid', subsample=(2,2),
                      dim_ordering='th', name='Conv3_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN3')(x)
    x = Convolution2D(512, 1, 1, init='he_normal', activation='relu', border_mode='valid', subsample=(1,1),
                      dim_ordering='th', name='1x1Conv2_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN4')(x)
    x = Convolution2D(512, 2, 2, init='he_normal', activation='relu', border_mode='valid', subsample=(2,2),
                      dim_ordering='th',name='Conv5_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN5')(x)
    x = Convolution2D(128, 1, 1, init='he_normal', activation='relu', border_mode='valid', subsample=(1,1),
                      dim_ordering='th', name='1x1Conv3_BC')(x)
    x = BatchNormalization(mode=2, axis=1, name='BC_BN6')(x)
    x = Flatten(name='BC_Flatten')(x)
    # x = BatchNormalization(mode=2, axis=1, name='BC_BN7')(x)
    # x = (Dense(8, init='he_normal', activation='relu', name='BC_FC1'))(x)
    predictions = Dense(14, init='zero', activation='softmax', name='BC_Softmax')(x)

    full_conv_model = Model(input=base_model.input, output=predictions)

    if load_weights == True:
        full_conv_model.load_weights(Conv_Test_Model_Weights_Path)

    return full_conv_model

def GetXYData(method_pp, test_sample_amt,classlist, dir_type): #only for the different test samples.
    method = method_pp # '', _PPd_G, _G Sobel, _PPd_Canny_Each, _PPd_Canny HSV

   # CAREFUL THIS IS NOT THE ORDER IN WHICH KERAS LOADS THE LABELS:
   # {'0_NoCar': 0, '3_Avanza': 9, '13_Civic': 3, '11_Jazz': 1, '22_XTrail': 7, '6_Corolla': 12, '21_Juke': 6,
   #            '5_Yaris': 11, '12_CRV': 2, '4_Kijang': 10, '9_Brio': 13, '2_Ayla': 8, '20_Livina': 5, '1_NotinDB': 4}
   #  KERAS LOADS IT LIKE THIS: ['0_NoCar','11_Jazz','12_CRV','13_Civic','1_NotinDB','20_Livina','21_Juke','22_XTrail','2_Ayla','3_Avanza','4_Kijang',
   #   '5_Yaris','6_Corolla','9_Brio']

    FolderNames = []
    FolderNames = os.listdir(Dataset_Path + str(dir_type) + str(method_pp) + '/')

    x_array=np.zeros((test_sample_amt,3,299,299),dtype=np.uint8)
    y_array=np.zeros((test_sample_amt),dtype=np.uint8)
    print(Dataset_Path + str(dir_type) + str(method_pp)+ '/' + str(dir_type) + '_Set_Labels.csv')
    with open(Dataset_Path + str(dir_type) + str(method_pp)+ '/' + str(dir_type) + '_Set_Labels.csv', 'rb') as csvin:
        crosswalk = csv.reader(csvin, dialect = 'excel')
        i=0
        # class_list = ['0_NoCar','1_NotinDB','2_Ayla','3_Avanza', '4_Kijang','5_Yaris',
        #  '6_Corolla','9_Brio','11_Jazz','12_CRV','13_Civic','20_Livina','21_Juke','22_XTrail']
        for row in crosswalk:
            filename=row[0]
            label=row[1]
            #print("Loaded: {} with label: {}".format(filename, label))
            img = io.imread(filename, plugin='matplotlib')
            x_array[i, 0, :, :] = img[:, :, 0]
            x_array[i, 1, :, :] = img[:, :, 1]
            x_array[i, 2, :, :] = img[:, :, 2]
            y_array[i] = classlist.index(label)
            #print ('y_array[i]: {}'.format(y_array[i]))
            #labels_list.append(label)
            i+=1
        #y_array = np.ndarray.flatten(y_array)
    csvin.close()
    return x_array, y_array

def GetDataRatio(method_pp,classlist,sample_amt,dir_type):

    classlist_as_ints=np.zeros((len(classlist)),dtype=np.uint8)
    labelcounter = 0
    for label_string in classlist:
        classlist_as_ints[labelcounter] = classlist.index(label_string)
        labelcounter += 1

    frequency_of_classes = np.zeros((len(classlist)), dtype=np.uint8)
    y_array=np.zeros((sample_amt),dtype=np.uint8)

    with open(Dataset_Path + str(dir_type) + str(method_pp)+ '/' + str(dir_type) + '_Set_Labels.csv', 'rb') as csvin:
        crosswalk = csv.reader(csvin, dialect = 'excel')
        i=0
        for row in crosswalk:
            filename = row[0]
            label = row[1]
            #print("Loaded: {} with label: {}".format(filename, label))
            y_array[i]=classlist.index(label) #the labels for each input data
            frequency_of_classes[classlist.index(label)] +=1
            i += 1
        #unique_indices_sorted = np.unique(y_array)
    csvin.close()

    print('freq of classes: {}'.format(frequency_of_classes))

    #class_weight_computed = compute_class_weight('balanced',classlist_as_ints,y_array)
    class_weight_computed = {}
    for i in classlist_as_ints:
        key=i
        class_weight_computed[key] = (float(frequency_of_classes[i])/float(sample_amt))
    print ('computed classweight: {}'.format(class_weight_computed))
    return class_weight_computed

if __name__ == '__main__':

    img_width=299
    img_height=299
    BATCH_SIZE = 42

    pp_method_list= ['_PPd_RGB', '_PPd_G', '_PPd_G_Canny', '_PPd_Sobel_Each','_PPd_Sobel_HSV']
    #pp_method_list = ['_PPd_Sobel_HSV']

    class_list = ['0_NoCar','11_Jazz','12_CRV','13_Civic','1_NotinDB','20_Livina','21_Juke','22_XTrail','2_Ayla','3_Avanza','4_Kijang',
     '5_Yaris','6_Corolla','9_Brio']

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

        full_model = BuildWithConvClassifier(Best_Model_Weights_Path, load_weights=False)

        full_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        full_model.summary()

        for layer in full_model.layers:
            layer.trainable = False  # freeze the entire inception model for now.

        #217 if you just want to train the top convnet #172 for mixed8 #158 for mixed 7
        for layer in full_model.layers[217:233]: #add +1 to the last layer index.. otherwise the sigmoid won't get trained!
            layer.trainable = True  # Train only the classifier

        for i, layer in enumerate(full_model.layers):
            print(i, layer.name, "Trainable?: ", layer.trainable)

        training_callbacks = [EarlyStopping(monitor='val_loss', patience=25, verbose=2),
                              ModelCheckpoint(filepath=Best_Model_Weights_Path,
                                              monitor='val_loss', verbose=2, save_best_only=True,
                                              save_weights_only=True)]

        class_weight_to_fit = GetDataRatio(ppmethod, class_list, nb_training_samples, 'Train')
        #print('class_weight_to_fit: {}'.format(class_weight_to_fit))

        hist=full_model.fit_generator(train_generator,samples_per_epoch=nb_training_samples,nb_epoch=1000,
                                      validation_data=validation_generator, nb_val_samples=42,
                                      callbacks=training_callbacks,
                                      class_weight=class_weight_to_fit)

        #class_weight=class_weight_to_fit

        best_epoch = np.argmin(np.asarray(hist.history['val_loss']))
        best_result = np.asarray((best_epoch,(np.asarray(hist.history['acc'])[best_epoch]),
                                 (np.asarray(hist.history['loss'])[best_epoch]),
                                (np.asarray(hist.history['val_acc'])[best_epoch]),
                                 (np.asarray(hist.history['val_loss'])[best_epoch])))
        print('best epoch index: {}, best result: {}'.format(best_epoch, best_result)) #actual epoch is index+1 because arrays start at 0..

        # # saves the best epoch's results
        np.savetxt(Base_Path + 'Results/BestEpochResult' + ppmethod + '.txt', best_result,
                   fmt='%5.6f', delimiter=' ', newline='\n', header='epoch, acc, loss, val_acc, val_loss',
                   footer=str(ppmethod), comments='# ')

        np.save(Base_Path + 'Results/acc' + ppmethod + '.npy', np.asarray(hist.history['acc']))
        np.save(Base_Path + 'Results/loss' + ppmethod + '.npy', np.asarray(hist.history['loss']))
        np.save(Base_Path + 'Results/val_acc' + ppmethod + '.npy', np.asarray(hist.history['val_acc']))
        np.save(Base_Path + 'Results/val_loss' + ppmethod + '.npy', np.asarray(hist.history['val_loss']))

        # # summarize history for accuracy
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model accuracy' + str(ppmethod))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Base_Path + 'Results/acc' + ppmethod + '.png', bbox_inches='tight')
        plt.clf()

        # # summarize history for loss
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss' + str(ppmethod))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Base_Path + 'Results/loss' + ppmethod + '.png', bbox_inches='tight')
        plt.clf()
        ## plt.show()

        #Now we test the best weight. "Instrumented best weight run"
        full_model.load_weights(Best_Model_Weights_Path) #LOAD (not set) the best weights.
        X_test, Y_test = GetXYData(ppmethod,nb_testing_samples,class_list,'Test') #get them big arrays
        Y_test = np.ndarray.flatten(Y_test)
        y_predictions = full_model.predict(X_test, batch_size=21)
        y_preds_as_classes = np_utils.categorical_probas_to_classes(y_predictions) #converts class probs to one-hot

        conf_matrix_testset = confusion_matrix(Y_test, y_preds_as_classes)
        np.savetxt(Base_Path + 'Results/CfMat_Test' + ppmethod + '.txt', conf_matrix_testset,
                   fmt='%1i', delimiter=' ', newline='\n', header='', footer='', comments='# ')
        plt.figure()
        plot_confusion_matrix(conf_matrix_testset, class_list, normalize=False,
                              title='Confusion Matrix Test' + str(ppmethod), cmap=plt.cm.Blues)
        plt.clf()


        #Now get the confusion matrix for the training set batch by batch..
        X_train_full, Y_train_full = GetXYData(ppmethod, nb_training_samples, class_list,'Train')
        # print('xtrainfull shape: {} ndim:{} ytrainfullshape: {} ndim {}'.format(X_train_full.shape, X_train_full.ndim,
        #                                                                         Y_train_full.shape, Y_train_full.ndim))
        #huge array and vector.. need to break into chunks.
        #there are 1738 training images. Training batch size is 21 (since test set is 42 large) -> only 1722 images used for this.

        train_conf_mat_batch_size = BATCH_SIZE
        X_train_chunk = np.zeros((train_conf_mat_batch_size,3,img_height,img_width),dtype=np.uint8)
        Y_train_chunk = np.zeros(train_conf_mat_batch_size)
        batch_counter = train_conf_mat_batch_size
        y_preds_initialized = False
        Y_train_cut = Y_train_full[0:-16] #1738 in total, deduct 16 values fron the right -> 1722.
        for i in range(0,1722-1): #len(Y_train_full)-
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
                    #print('appended')

        conf_matrix_trainset = confusion_matrix(Y_train_cut, y_preds_full)
        plot_confusion_matrix(conf_matrix_trainset, class_list, normalize=False,
                              title='Confusion Matrix Train' + str(ppmethod), cmap=plt.cm.PuRd)
        plt.clf()






