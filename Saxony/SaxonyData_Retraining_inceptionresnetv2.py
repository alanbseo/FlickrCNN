
#### Fine-tune InceptionV4 on a new set of classes

# The task of fine-tuning a network is to tweak the parameters of an already trained network so that it adapts to the new task at hand. As explained here, the initial layers learn very general features and as we go higher up the network, the layers tend to learn patterns more specific to the task it is being trained on. Thus, for fine-tuning, we want to keep the initial layers intact ( or freeze them ) and retrain the later layers for our task.
# Thus, fine-tuning avoids both the limitations discussed above.
#
# The amount of data required for training is not much because of two reasons. First, we are not training the entire network. Second, the part that is being trained is not trained from scratch.
# Since the parameters that need to be updated is less, the amount of time needed will also be less.
#


# Ref:
# https://github.com/fchollet/deep-learning-with-python-notebooks
# https://gist.github.com/liudanking
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
# https://github.com/jkjung-avt/keras-cats-dogs-tutorial/blob/master/train_inceptionresnetv2.py
# https://forums.fast.ai/t/globalaveragepooling2d-use/8358

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



default_path = '/home/alan/Dropbox/KIT/FlickrEU/FlickrCNN'
# default_path = '/Users/seo-b/Dropbox/KIT/FlickrEU/FlickrCNN'

os.chdir(default_path)
# photo_path = default_path + '/Photos_168_retraining'


from keras.applications import inception_resnet_v2


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam
from keras import metrics

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


img_width, img_height = 662, 662
# train_data_dir = "Photos_338_retraining/train"
# validation_data_dir = "Photos_338_retraining/validation"
# nb_train_samples = 210
# nb_validation_samples = 99

# train_data_dir = "Photos_338_retraining_wovalidation/train"
# validation_data_dir = "Photos_338_retraining_wovalidation/validation"

train_data_dir = "Saxony_Flickr_Selection_190424/training"
validation_data_dir = "Saxony_Flickr_Selection_190424/validation"

nb_train_samples = 255
nb_validation_samples = 0

batch_size = 32 #  Means the number of images used in one batch. If you have 320 images and your batch size is 32, you need 10 internal iterations go through the data set once (which is called `one epoch')
# It is set  proportional to the training sample size. There are discussions but generally if you can afford, bigger is better. It

epochs = 100 # An epoch means the whole input dataset has been used for training the network. There are some heuristics to determine the maximum epoch. Also there is a way to stop the training based on the performance (callled  `Early stopping').

num_classes = 9




##### build our classifier model based on pre-trained InceptionResNetV2:


# Load the base pre-trained model

# do not include the top fully-connected layer
# 1. we don't include the top (fully connected) layers of InceptionResNetV2

model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
# Freeze the layers which you don't want to train. Here I am freezing the all layers.
# i.e. freeze all InceptionV3 layers




# New dataset is small and similar to original dataset:
# There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
# So lets freeze all the layers and train only the classifier

# first: train only the top layers

# for layer in net_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in net_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True

x = model.output

# Now that we have set the trainable parameters of our base network, we would like to add a classifier on top of the convolutional base. We will simply add a fully connected layer followed by a softmax layer with num_classes outputs.

# Adding custom Layer
# x = Flatten()(x)
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)


# If the network is stuck at 50% accuracy, there’s no reason to do any dropout. Dropout is a regularization process to avoid overfitting. But your problem is underfitting.
# x = Dropout(0.5)(x) # 50% dropout

# A Dense (fully connected) layer which generates softmax class score for each class
predictions = Dense(num_classes, activation='softmax', name='softmax')(x)



# creating the final model
# this is the model we will train
model_final = Model(inputs = model.input, outputs = predictions)



#Now we will be training only the classifiers (FC layers)

# compile the model (should be done *after* setting layers to non-trainable)

#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# model_final.compile(optimizer='rmsprop', loss='categorical_crossentropy')


## load previously trained weights
model_final.load_weights('TrainedWeights/InceptionResnetV2_Saxony_retrain_flickr_9classes_epoch60_acc0.98.h5')

FREEZE_LAYERS = len(model_final.layers) - 3 # train only last few layers

for layer in model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False


# Compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
# For classification, categorical_crossentropy is most often used. It measures the information between the predicted and the true class labels similarly with the mutual information. It is
model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


print(model_final.summary())




# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)
test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")



# show class indices
print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')



# Save the model according to the conditions
checkpoint = ModelCheckpoint("TrainedWeights/InceptionResnetV2_Saxony_retrain.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=5)

# Setup the early stopping criteria
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')



# Re-train the model
history = model_final.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples,
    epochs = epochs,
 #   validation_data = validation_generator,
 #   validation_steps = nb_validation_samples,
    callbacks = [checkpoint, early])

# at this point, the top layers are well trained.



# Save the model
model_final.save('TrainedWeights/InceptionResnetV2_Saxony_retrain_flickr_9classes_epoch30_acc0.98.h5')


# Save the model architecture
with open('InceptionResnetV2_Saxony_retrain_flickr_final_architecture.json', 'w') as f:
    f.write(model_final.to_json())



acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot( acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()



# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

#errors = np.where(predicted_classes != ground_truth)[0]
errors = np.where(np.not_equal(predicted_classes, ground_truth[0]))

print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()


####
### @todo feature extraction



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



