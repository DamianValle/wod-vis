import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.metrics import accuracy_score
from vis.visualization import visualize_cam
import keras.backend as K
import sys
import json
from keras.applications.vgg16 import preprocess_input
from PIL import Image

def rgb2gray(rgb):
    x = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    x = x/255
    return np.expand_dims(x, axis=2)

print('Loading model...')
model = keras.models.load_model('models/arrow.h5')
print(model.summary())

print('Reading files...')
with open('X_train_arrow.txt', 'r') as filehandle:
    X_train = json.load(filehandle)
with open('y_train_arrow.txt', 'r') as filehandle:
    y_train = json.load(filehandle)
X_train = np.array(X_train)
y_train = np.array(y_train)

last_conv_layer = model.get_layer('conv2d_8')

imgs = np.array([rgb2gray(np.array(Image.open('/work3/s182091/rico/combined/{}.jpg'.format(photo_path)).resize((360, 640)))) for photo_path in X_train[:50]])
y_train = y_train[:50]

for imindex in range(30):

    cvimg = cv2.imread('/work3/s182091/rico/combined/{}.jpg'.format(str(X_train[imindex])))
    x = np.expand_dims(imgs[imindex], axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(64):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    out=heatmap
    np.divide(heatmap, np.max(heatmap), out=out, where=np.max(heatmap)!=0)
    heatmap=out

    heatmap = cv2.resize(heatmap, (cvimg.shape[1], cvimg.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #cv2.imwrite('figs/{}_pred{}.png'.format(imindex, class_idx), superimposed_img)

    heatgray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(heatgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    im, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour

    x, y, w, h = cv2.boundingRect(maxContourData)

    mask = np.zeros_like(th2)
    cv2.fillPoly(mask,[maxContourData],1)

    R,G,B = cv2.split(cvimg)
    finalImage = np.zeros_like(cvimg)
    finalImage[:,:,0] = np.multiply(R,mask)
    finalImage[:,:,1] = np.multiply(G,mask)
    finalImage[:,:,2] = np.multiply(B,mask)

    cv2.rectangle(finalImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('figs/{}original_pred{}.png'.format(imindex, class_idx), cvimg)

    cv2.imwrite('figs/{}heat_pred{}.png'.format(imindex, class_idx), heatmap)

    cv2.imwrite('figs/{}thres_pred{}.png'.format(imindex, class_idx), th2)

    th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(th2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('figs/{}thresbb_pred{}.png'.format(imindex, class_idx), th2)

    cv2.rectangle(cvimg, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('figs/{}final_pred{}.png'.format(imindex, class_idx), cvimg)












