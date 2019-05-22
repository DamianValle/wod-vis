import os
import json
import numpy as np
import PIL
from PIL import Image

import keras
from keras.utils import Sequence
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.utils import Sequence
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


batch_size = 16

with open('X_train_arrow.txt', 'r') as filehandle:  
    X_train = json.load(filehandle)
with open('y_train_arrow.txt', 'r') as filehandle:  
    y_train = json.load(filehandle)

class Batch_gen(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.array(Image.open('/work3/s182091/rico/combined/{}.jpg'.format(photo_path)).resize((360, 640)))/255 for photo_path in batch_x]), np.array(batch_y)

y_train = keras.utils.to_categorical(y_train)
train_batchgen = Batch_gen(X_train, y_train, batch_size)


###################################
####     VGG archictecture     ####
###################################

vgg_16_model = VGG16(weights='imagenet', include_top=False, input_shape = (640, 360, 3))

# get layers and add average pooling layer
x = vgg_16_model.output
x = GlobalAveragePooling2D()(x)
#x = Flatten()(x)

# add fully-connected layers
x = Dense(300, activation='relu')(x)
x = Dense(20, activation='relu')(x)

# add output layer
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=vgg_16_model.input, outputs=predictions)

#for layer in vgg_16_model.layers:
    #layer.trainable = False

print(model.summary())

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batchgen,
              steps_per_epoch=(20000 // batch_size),
              shuffle = True,
              epochs=10,
              use_multiprocessing=True,
              callbacks=[tensorboard])

model.save("models/vgg_16.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig("acc2-fine.png")












