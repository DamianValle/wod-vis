import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import pandas as pd
from keras.optimizers import SGD
import keras
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.utils import Sequence
import time

class Batch_gen(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([mpimg.imread(('/zhome/12/f/134534/Downloads/processedImage/' + photo_path)) for photo_path in batch_x]), np.array(batch_y)

seq_file = open('XMLsequence.lst')
train_file = open('train.lst')
val_file = open('validate.lst')

count = 0
for line in train_file:
    count = count + 1
train_file.close()
train_file = open('train.lst')

seq_lines = seq_file.readlines()

test_file = open('test_shuffle.lst')




train_paths = []
train_labels = []
val_paths = []
val_labels = []

for line in train_file:
    sp = line.split(" ")
    photo_path = sp[0]
    idx = int(sp[1])
    
    train_paths.append(photo_path)
    train_labels.append(1 if "android.widget.ImageButton" in seq_lines[idx] else 0)

for line in val_file:
    sp = line.split(" ")
    photo_path = sp[0]
    idx = int(sp[1])
    
    val_paths.append(photo_path)
    val_labels.append(1 if "android.widget.ImageButton" in seq_lines[idx] else 0)

print(np.shape(train_paths))
print(count)

train_labels = keras.utils.to_categorical(train_labels)
val_labels = keras.utils.to_categorical(val_labels)

time.sleep(5)

train_batchgen = Batch_gen(train_paths, train_labels, 128)
val_batchgen = Batch_gen(val_paths, val_labels, 128)


##################################3

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(201,301,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batchgen,
              validation_data = val_batchgen,
              steps_per_epoch=(count // 128),
              epochs=10,
              use_multiprocessing=True)

model.save("cnn-imagebutton-10ep-128bs.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig("acc2.png")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("loss2.png")

