# Convolutions

# import
import tensorflow as tf
print(tf.__version__)

# load dataset
mnist = tf.keras.datasets.fashion_mnist

# load training and test
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# reshape
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# define a model with Convolutions
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# summary
model.summary()

# fit a model
model.fit(training_images, training_labels, epochs=20)

# evaluate
results = model.evaluate(test_images, test_labels)
print(results)