import os
import json
import numpy as np
import PIL
from PIL import Image

import keras
from keras.utils import Sequence
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.models import Sequential, Model, load_model
from keras.utils import Sequence
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [tensorboard]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


os.chdir("/zhome/12/f/134534/Desktop/rico")

def rgb2gray(rgb):
    x = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    x = x/255
    return np.expand_dims(x, axis=2)

##########################
####    Dataloader    ####
##########################

with open('X_train_fav.txt', 'r') as filehandle:  
    X_train = json.load(filehandle)
with open('y_train_fav.txt', 'r') as filehandle:  
    y_train = json.load(filehandle)
X_train = np.array(X_train)
y_train = np.array(y_train)

y_train = keras.utils.to_categorical(y_train)

img_train = np.array([rgb2gray(np.array(Image.open('/work3/s182091/rico/combined/{}.jpg'.format(photo_path)).resize((360, 640)))) for photo_path in X_train])

###################################
####     CNN archictecture     ####
###################################

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(640, 360, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

print(model.summary())

#uncomment for fine tuning
#model = load_model('models/arrow.h5')

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(img_train, y_train,
              batch_size = 32,
              validation_split = 0.1,
              shuffle = True,
              epochs=40,
              callbacks = callbacks_list)

model.save("models/arrow.h5")

plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("acc.png")

# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.png")

















    













