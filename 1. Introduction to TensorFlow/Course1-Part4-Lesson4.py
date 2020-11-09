
# import
import tensorflow as tf

# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') > 0.95):
            print('\nReached 95% accuracy so cancelling training!')
            self.model.stop_training = True

# reference callbacks
callbacks = myCallback()

# load dataset
mnist = tf.keras.datasets.fashion_mnist

# load training and test datasets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalization
training_images = training_images / 255.0
test_image = test_images / 255.0

# define a model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile a model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit a model
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

# evaluate a model
results = model.evaluate(test_images, test_labels)
print('results: ' + str(results))

# predict
classifications = model.predict(test_images)

print(classifications[1])
print(test_labels[1])

print(classifications[100])
print(test_labels[100])

print(classifications[50])
print(test_labels[50])

print(classifications[200])
print(test_labels[200])

