import keras
import numpy as np
import os
import cv2

import csv
import pandas as pd
import pathlib
import fnmatch


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


from keras.applications import inception_resnet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


from keras.preprocessing import image
from keras.applications import vgg16
import numpy as np


# or qt5.
# PyQt5 or similar required.. (https://stackoverflow.com/questions/52346254/importerror-failed-to-import-any-qt-binding-python-tensorflow)




default_path = '/Users/seo-b/Dropbox/KIT/Peatland/NATCAP'
os.chdir(default_path)
photo_path = default_path + '/Photos_50'

### Read filenames
filenames = os.listdir(photo_path)

filenames1 = fnmatch.filter(filenames, "*.jpg")
filenames2 = fnmatch.filter(filenames, "*.JPG")

filenames = filenames1 + filenames2


##### Predict

from keras import backend as K

#Load the Inception_V4_resnetv2 model
inceptionResnet_v2_model = inception_resnet_v2.InceptionResNetV2( weights='imagenet')
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

modelname = "InceptionResnetV2"
dataname = "Photos_50"


# filename = 'photoid_19568808955.jpg' # granpa
# filename = 'photoid_23663993529.jpg' # bridge


for filename in filenames:


    fname = photo_path + "/" + filename

    if os.path.isfile(fname):

        # load an image in PIL format
        # original = load_img(filename, target_size=(299, 299))
        original = load_img(fname, target_size=(331, 331))

        # convert the PIL image to a numpy array
        # IN PIL - image is in (width, height, channel)
        # In Numpy - image is in (height, width, channel)
        numpy_image = img_to_array(original)

        # Convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # Thus we add the extra dimension to the axis 0.
        image_batch = np.expand_dims(numpy_image, axis=0)


        # prepare the image (normalisation for channels)
        processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())



        # get the predicted probabilities for each class
        predictions = inceptionResnet_v2_model.predict(processed_image)
        # print predictions

        # convert the probabilities to class labels
        # We will get top 5 predictions which is the default
        predicted_tags = decode_predictions(predictions,  top=10)
        print('Predicted:', predicted_tags )


        df = pd.DataFrame(predicted_tags[0])
        name_csv = default_path + "/Result/Tag_" + modelname + "/" + filename + ".csv"



        # df.to_csv(name_csv)
        header = ["Rank", "ConceptID", "Concept", "PhotoID"]

        df.to_csv(name_csv, index_label=header)



        # Heatmaap

        # `img` is a PIL image of size 224x224
        img = image.load_img(fname, target_size=(224, 224))

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        x = vgg16.preprocess_input(x)

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
        # plt.matshow(heatmap)
        # plt.show()


        # We use cv2 to load the original image
        img = cv2.imread(fname)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img


        ## @todo vgg to incresv2
        # Save the image to disk
        cv2.imwrite("Result/Heatmap_" + modelname + "/AttentionMap_" + df[1][0] + "_" + filename, superimposed_img)



