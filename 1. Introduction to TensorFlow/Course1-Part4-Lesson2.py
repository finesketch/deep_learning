# Beyond Hello World, A Computer Vision Example
# =============================================

import tensorflow as tf
print(tf.__version__)


# Fashion MNIST
# =============

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

print(training_images.shape)
print(training_labels.shape)
print(test_images.shape)
print(test_labels.shape)

import numpy as np
np.set_printoptions(linewidth=150)

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


# Normalization
# =============

training_images = training_images / 255.0
test_images = test_images / 255.0


# Define, Compile, Fit, and Evaluate the Neutral Network
# ======================================

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

results = model.evaluate(test_images, test_labels)

print('Model Evaluation: ' + str(results))



# Exploration Exercises - 1
# =========================

# Predict
# =======

classifications = model.predict(test_images)
print(classifications[0])

# Verify
# ======
print(test_labels[0])


# Exploration Exercises - 2 (MNIST) (More neurons)
# =========================

print('Exploration Exercises - 2 (MNIST)')

# import
import tensorflow as tf
print(tf.__version__)

# load MNIST datasets
mnist = tf.keras.datasets.mnist

# load training/test data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define a model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile a model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

# Fit a model
model.fit(training_images, training_labels, epochs=5)

# Evaluate a model
results = model.evaluate(test_images, test_labels)
print('Evaluate a model: ' + str(results))

# Predict a output
classifications = model.predict(test_images)
print('classifications: ' + str(classifications[0]))
print('test_labels:' + str(test_labels[0]))


# Exploration Exercises - 3 (MNIST) (More Layers)
# =================================

# Import
import tensorflow as tf
print(tf.__version__)

# load MNIST database
mnist = tf.keras.datasets.mnist

# load training/testing data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define a model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')

# Fit a model
model.fit(training_images, training_labels)

# Evaluate a model
results = model.evaluate(test_images, test_labels)
print('Evaluate a model: ' + str(results))

# Predict a value
classifications = model.predict(test_images)
print('classifications: ' + str(classifications[0]))
print('test_labels: ' + str(test_labels[0]))



# Exploration Exercises - 6 (MNIST) (More epochs)
# =================================

# Import
import tensorflow as tf
print(tf.__version__)

# load dataset
mnist = tf.keras.datasets.mnist

# load training and test dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define a model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')

# Fit a model
model.fit(training_images, training_labels, epochs=30)

# Evaluate a model
results = model.evaluate(test_images, test_labels)
print('results: ' + str(results))

# Predict
classifications = model.predict(test_images)

print(classifications[34])
print(test_labels[34])


# Exploration Exercises - 6 (FahioMNIST) (Callback)
# =================================================

# import
import tensorflow as tf
print(tf.__version__)

# Define Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print('\nReached 60% accuracy so cancelling training!')
            self.model.stop_training = True

# setup callback
callbacks = myCallback()

# load database
mnist = tf.keras.datasets.fashion_mnist

# load training and test datasets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# define a model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')

# fit a model
model.fit(training_images, training_labels, epochs=30, callbacks=[callbacks])

# evaluate a model
results = model.evaluate(test_images, test_labels)
print('evaluation: ' + str(results))

# predict
classifications = model.predict(test_images)

# check
print(classifications[0])
print(test_labels[0])



