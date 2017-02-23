'''Visualization of the filters of Inception, via gradient ascent in input space.

'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.models import Model
from keras.layers import Convolution2D, Input
import Inception_PCL as ip
#import NotSoDeep_PCL_ConvNet as nsd
#import Inception_PCL_Module8 as mod8

#
from keras import backend as K

K.set_image_dim_ordering('th')

# dimensions of the generated pictures for each filter.
img_width = 512
img_height = 512

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)

#list of layers = ['Conv2_BC','Conv3_BC','Conv5_BC']
#layer dims = [1024, 1024, 512]
layer_name = ('Conv2_BC')

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#WeightsToLoad = []
# build the VGG16 network with ImageNet weights
Base_Path = '/home/ihsan/Documents/MIK/PCL_Project/'
Best_Model_Weights_Path=Base_Path + "Weights/Best_Full_Conv_Model.h5"
#Best_Model_Weights_Path=Base_Path + "Weights/Imagenet_NotSoDeepConv_Model_Mixed1.h5"

# Layer_Weight_Path = Base_Path + '/Weights/Conv2.npy' #[0]
# Bias_Weight_Path = Base_Path + '/Weights/Conv2_Bias.npy' #[1]
#BC_BN1 (BatchNormalization)      (None, 1024, 8, 8)    2048        1x1Conv1_BC[0][0]
#x = Convolution2D(1024,2,2,init='he_normal',activation='relu',border_mode='valid',subsample=(2,2),dim_ordering='th',name='Conv2_BC')(x)
# a = Input(shape=(1024,8,8))
# b = Convolution2D(1024,2,2,init='he_normal',activation='relu',border_mode='valid',subsample=(2,2),dim_ordering='th',name='Conv2_BC')(a)
#
# model = Model(input=a, output=b)

# Layer_Weight = np.load(Layer_Weight_Path)
# Bias = np.load(Bias_Weight_Path)
# model.get_layer("Conv2_BC").set_weights([Layer_Weight, Bias])
model = ip.BuildWithConvClassifier(Best_Model_Weights_Path, load_weights=True)
#model = mod8.BuildWithConvClassifier(Best_Model_Weights_Path, load_weights=True)
#model=nsd.BuildNotSoDeepConvNet(Best_Model_Weights_Path, load_weights=True)
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


#kept_filter_last_time=(94,140,329,519,613,627,56,91,150,209,259,272,314,419,424,447,516,536,551,748,762,871,924,911,928,956,962,944,968,993,996,935,908,904,850,750)
#kept_filter_last_time=(94,140,329,519,613,627,56,91,150,209)
kept_filter_last_time=(255,258,269,274,275,281,283,296)
kept_filters = []
for filter_index in range(100,106):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_dim_ordering() == 'th':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1

    # we start from a gray image with some random noise
    if K.image_dim_ordering() == 'th':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(11000):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Iteration %d, current loss value: %d', i, loss_value)
        if loss_value <= 1e-5:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
        #print(kept_filters(0),kept_filters(1))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = 2

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        print('kept filter index being analyzed: ', i*n+j)
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave('stitched_filters_%dx%d_Mixed10Conv2_1.png' % (n, n), stitched_filters)

