
#### Fine-tune InceptionV3 on a new set of classes

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

#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
os.environ['HIP_VISIBLE_DEVICES'] = '0,1'


import csv
import pandas as pd
import pathlib
import fnmatch

# split utils from the web
import split_utils


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


from tensorflow.python.client import device_lib



from keras.preprocessing import image
from keras.applications import vgg16
import numpy as np

from tensorflow.keras.utils import multi_gpu_model # Multi-GPU Training ref: https://gist.github.com/mattiavarile/223d9c13c9f1919abe9a77931a4ab6c1



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


img_width, img_height = 331, 331
# train_data_dir = "Photos_338_retraining/train"
# validation_data_dir = "Photos_338_retraining/validation"
# nb_train_samples = 210
# nb_validation_samples = 99

train_data_dir = "Photos_iterative_Aug2019/train"
validation_data_dir = "Photos_iterative_Aug2019/validation"

batch_size = 64 # proportional to the training sample size.. (64 did not work for Vega56 8GB, 128 did not work for Radeon7 16GB)
epochs = 100

num_classes = 16




##### build our classifier model based on pre-trained InceptionResNetV2:


# Load the base pre-trained model

# do not include the top fully-connected layer
# 1. we don't include the top (fully connected) layers of InceptionResNetV2

model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
# Freeze the layers which you don't want to train. Here I am freezing the all layers.
# i.e. freeze all InceptionV3 layers
# model.aux_logits=False


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

# https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
# add a global spatial average pooling layer
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1312.4400.pdf
# allows you to have the input image be any size, not just a fixed size like 227x227.
# It does through taking an average of every incoming feature map.
# For example, with a 15x15x8 incoming tensor of feature maps, we take the average of each 15x15 matrix slice, giving us an 8 dimensional vector.
# We can now feed this into the fully connected layers.
# Notice how that the size of the matrix slices can change, for example, the input might be 32x32x8,
# and we’ll still get an 8 dimensional vector as an output from the global average pooling layer.
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
# https://datascience.stackexchange.com/questions/28120/globalaveragepooling2d-in-inception-v3-example
# If the network is stuck at 50% accuracy, there’s no reason to do any dropout.
# Dropout is a regularization process to avoid overfitting. But your problem is underfitting.
x = Dropout(0.3)(x) # 30% dropout

# A Dense (fully connected) layer which generates softmax class score for each class
predictions = Dense(num_classes, activation='softmax', name='softmax')(x)


# creating the final model
# this is the model we will train
model_final = Model(inputs = model.input, outputs = predictions)


#Now we will be training only the classifiers (FC layers)


## load previously trained weights
# model_final.load_weights('TrainedWeights/InceptionResnetV2_retrain_instagram_epoch150_acc0.97.h5')
model_final.load_weights('TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_bigdata_epoch100_val_acc0.88.h5')


# First retraining
#for layer in model.layers[:]:
#    layer.trainable = False


# We can start fine-tuning convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model_final.layers):
    print(i, layer.name)

# Fine tuning (
FREEZE_LAYERS = len(model.layers) - 5 # train the newly added layers and the last few layers

for layer in model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False



# @todo multi gpu throws an error possibly due to version conflicts..
# model_final = multi_gpu_model(model_final, gpus=2, cpu_merge=True, cpu_relocation=False)

# compile the model (should be done *after* setting layers to non-trainable)


# we need to recompile the model for these modifications to take effect
# Compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
#model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy', 'loss', 'val_acc'])
model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# we can use SGD with a low learning rate
#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])

print(model_final.summary())

# Save the model architecture
with open('InceptionResnetV2_retrain_instagram_final_architecture.json', 'w') as f:
    f.write(model_final.to_json())

validation_split = 0.3

# all data in train_dir and val_dir which are alias to original_data. (both dir is temporary directory)
# don't clear base_dir, because this directory holds on temp directory.
base_dir, train_tmp_dir, val_tmp_dir = split_utils.train_valid_split(train_data_dir, validation_split, seed=1)




# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


# generator for validation data
val_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


train_generator = train_datagen.flow_from_directory(
     train_tmp_dir,
     target_size = (img_height, img_width),
     batch_size = batch_size,
     class_mode = "categorical")

val_batch_size = batch_size

validation_generator = val_datagen.flow_from_directory(
    val_tmp_dir,
    target_size = (img_height, img_width),
    batch_size=val_batch_size,
    class_mode = "categorical")



# show class indices
print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')



nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n

print('the ratio of validation_split is {}'.format(validation_split))
print('the size of train_dir is {}'.format(nb_train_samples))
print('the size of val_dir is {}'.format(nb_validation_samples))


# Save the model according to the conditions
checkpoint = ModelCheckpoint("TrainedWeights/InceptionResnetV2_Seattle_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=3)


early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# steps per epoch depends on the batch size
steps_per_epoch = int(np.ceil(nb_train_samples / batch_size))
validation_steps_per_epoch = int(np.ceil(nb_validation_samples / batch_size))

# Tensorboard
# If printing histograms, validation_data must be provided, and cannot be a generator.
callback_tb = keras.callbacks.TensorBoard(
        log_dir = "log_dir", # tensorflow log
        histogram_freq=1,    # histogram
        # embeddings_freq=1,
        # embeddings_data=train_generator.labels,
        write_graph=True, write_images=True
    )

# callback_tb_simple = keras.callbacks.TensorBoard(
#         log_dir = "log_dir", # tensorflow log
#         # histogram_freq=1,    # histogram
#         # embeddings_freq=1,
#         # embeddings_data=train_generator.labels,
#         write_graph=True, write_images=True
#     )


# Train the model

history = model_final.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps_per_epoch,
    callbacks = [checkpoint, early]) # , callback_tb



# at this point, the unfreezed layers are well trained.

# Save the model
model_final.save('TrainedWeights/InceptionResnetV2_retrain_instagram_final.h5')

# save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('TrainedWeights/InceptionResnetV2_retrain_instagram_final.csv')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot( acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()



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



### Feature extraction



####
### @todo feature extraction



#





















