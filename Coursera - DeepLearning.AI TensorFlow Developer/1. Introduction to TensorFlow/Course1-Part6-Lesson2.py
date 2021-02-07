# Improving Computer Vision Accuracy using Convolutions

# import
import tensorflow as tf
print(tf.__version__)
from os import path, getcwd, chdir

# load dataset
mnist = tf.keras.datasets.fashion_mnist

# load training and test datasets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# define a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit a model
model.fit(training_images, training_labels, epochs=5)

# evaluate a model
results = model.evaluate(test_images, test_labels)
print(results)