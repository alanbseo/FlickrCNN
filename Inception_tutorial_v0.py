
###
### You might consider running this script within a virtual environment
###  like this, for example, from the command line:

## first, setup a virtualenv from shell...
# virtualenv -p python3 venv_activities
# source venv_activities/bin/activate

## with the right packages...
# pip install tensorflow
# pip install keras
# pip install opencv-python
# pip install requests
# pip install matplotlib

## then launch python(3)...
# python






import keras
import numpy as np
import os
import cv2

from keras.applications import inception_v3, inception_resnet_v2 # , nasnet, xception


import ssl

### Avoid certificat error (source: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)
import requests
requests.packages.urllib3.disable_warnings()

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context



#Load the Inception_V3 model
# inception_v3_model = inception_v3.InceptionV3(weights='imagenet')

#Load the Inception_V4_resnetv2 model
inceptionResnet_v2_model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# inceptionResnet_v2_model = inception_resnet_v2.InceptionResNetV2( weights='imagenet')
# nasnet_model = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# xception_model = xception.Xception(include_top=True, weights='imagenet')


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
%matplotlib osx
# or qt5.
# PyQt5 or similar required.. (https://stackoverflow.com/questions/52346254/importerror-failed-to-import-any-qt-binding-python-tensorflow)



default_path = '/Users/seo-b/Dropbox/KIT/Peatland/NATCAP'
os.chdir(default_path)

# filename = 'Photos_50/photoid_19568808955.jpg' # granpa
filename = 'Photos_50/photoid_23663993529.jpg' # bridge

# load an image in PIL format
# original = load_img(filename, target_size=(299, 299))
original = load_img(filename, target_size=(331, 331))

# Typical input image sizes to a Convolutional Neural Network trained on ImageNet are 224×224, 227×227, 256×256, and 299×299; however, you may see other dimensions as well.
# VGG16, VGG19, and ResNet all accept 224×224 input images while Inception V3 and Xception require 299×299 pixel inputs, as demonstrated by the following code block:
# Nasnet large 331x331

print('PIL image size', original.size)
plt.imshow(original)
plt.show()



from PIL import Image, ImageDraw

# ImageDraw.Draw(
#     original  # Image
# ).text(
#     (0, 0),  # Coordinates
#     ''.join(original.size.__str__()),  # Text
#     (0, 0, 0)  # Color
# )
# plt.imshow(original)


d = ImageDraw.Draw(original)
d.text((10,10), ''.join(original.size.__str__()), col=(0, 0, 0), fill=(255,255,0))
# d.text((10,10), ''.join(original.size.__str__()), col=(0, 0, 0), fill=(255,255,0))

# d.draw()
plt.imshow(original)
plt.show()


# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)

# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))



# prepare the image for the VGG model (normalisation for channels)
processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())



# get the predicted probabilities for each class
predictions = inceptionResnet_v2_model.predict(processed_image)
# print predictions

# convert the probabilities to class labels
# We will get top 5 predictions which is the default
predicted_tags = decode_predictions(predictions,  top=10)
print('Predicted:', predicted_tags )






#
# # get the predicted probabilities for each class
# predictions = nasnet_model.predict(processed_image)
# # print predictions
#
# # convert the probabilities to class labels
# # We will get top 5 predictions which is the default
# print('Predicted:', decode_predictions(predictions, top=10    ))
#
#
#
# # get the predicted probabilities for each class
# predictions = inception_v3_model.predict(processed_image)
# # print predictions
#
# # convert the probabilities to class labels
# # We will get top 5 predictions which is the default
# print('Predicted:', decode_predictions(predictions, top=10))
#
#
# # get the predicted probabilities for each class
# predictions = nasnet_model.predict(nasnet.preprocess_input(image_batch.copy()))
#
# # print predictions
#
# # convert the probabilities to class labels
# # We will get top 5 predictions which is the default
# print('Predicted:', decode_predictions(predictions,  top=10))
#
#



##### Heatmap (source: http://fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb#Visualizing-heatmaps-of-class-activation)

from keras.preprocessing import image
from keras.applications import vgg16
import numpy as np

from keras import backend as K

# `img` is a PIL image of size 224x224
img = image.load_img(filename, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = vgg16.preprocess_input(x)
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')


preds = vgg_model.predict(x)
print('Predicted:', decode_predictions(preds, top=10)[0])

dominant_feature_idx = np.argmax(preds[0])


# This is the dominant entry in the prediction vector
dominant_output = vgg_model.output[:, dominant_feature_idx]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = vgg_model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(dominant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([vgg_model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)


# For visualization purpose, we will also normalize the heatmap between 0 and 1:

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()




# We use cv2 to load the original image
img = cv2.imread(filename)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img
decode_predictions(preds, top=10)[0]



# Save the image to disk
cv2.imwrite('heatmap_superimposed.jpg', superimposed_img)





####
### @todo feature extraction, fine tuning, network structure modification..


# Fine-tune InceptionV3 on a new set of classes

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adam


batch_size = 50
num_classes = 3
n_epoch = 10


# create the base pre-trained model
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have n classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# New dataset is small and similar to original dataset:
# There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
# So lets freeze all the layers and train only the classifier
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers:
    layer.trainable = False
#Now we will be training only the classifiers (FC layers)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=Adam(lr=0.0001),
#               metrics=['acc'])


### @todo feed new data here
x_train = np.random.normal(loc=127, scale=127, size=(50, 224,224,3))
y_train = np.array([0,1]*25)
x_train = inception_v3.preprocess_input(x_train)

print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0))

# train the model on the new data for a few epochs
model.fit(x_train, y_train,
          epochs=n_epoch,
          batch_size=batch_size,
          shuffle=False,
          validation_data=(x_train, y_train))


# at this point, the top layers are well trained.
#
# We can start fine-tuning convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# model.fit_generator(...)

### Feature extraction


