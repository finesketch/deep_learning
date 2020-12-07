import numpy as np
from keras.preprocessing import image
import os, os.path

# load images from a folder
images = []
path = '/tmp/horse-or-human/horses'
valid_images = ['.jpg', '.png', '.gif']

for fn in os.listdir(path):
    ext = os.path.splitext(fn)[1]
    if ext.lower() not in valid_images:
        continue
    images.append(os.path.join(path, fn))
    print(fn)

