# unzip

import os
import zipfile

#local_zip = '/tmp/horse-or-human.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('/tmp/horse-or-human')
#local_zip = '/tmp/validation-horse-or-human.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('/tmp/validation-horse-or-human')
#zip_ref.close()

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horse_names = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)


# building a small model from scratch

import tensorflow as tf

# create a model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# summary

model.summary()

# compile a model

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

# data preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(150,150),
    batch_size=128,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    '/tmp/validation-horse-or-human/',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

# fit a model

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8)


# predict the results

import numpy as np
from keras.preprocessing import image
import os, os.path

# load images from a folder
images = []
#path = '/tmp/horse-or-human-test'
path = '/tmp/horse-or-human-test'
valid_images = ['.jpg', '.png', '.gif']

for fn in os.listdir(path):
    ext = os.path.splitext(fn)[1]
    if ext.lower() not in valid_images:
        continue
    images.append(os.path.join(path, fn))
    img = image.load_img(os.path.join(path, fn), target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_vstack = np.vstack([x])
    classes = model.predict(image_vstack, batch_size=10)
    if classes[0] > 0.5:
        print(fn + ' is a human (' + str(classes[0]) + ')')
    else:
        print(fn + ' is a horse (' + str(classes[0]) + ')')





