import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import pickle

#testY is the labels
#trainX is the images

names = ['DC','Marvel']

pickle_in = open("trainX.pickle","rb")
trainX = pickle.load(pickle_in)


pickle_in = open("testY.pickle","rb")
testY = pickle.load(pickle_in)

trainX = trainX/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=trainX.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainX, testY, batch_size=2, epochs=5, validation_split=0.3)

model.evaluate(trainX,testY)

predictions = model.predict(trainX)
plt.figure(figsize=(150,150))

trainX = trainX.squeeze()



for i in range(20):
    plt.grid(False)
    dum = predictions[i]
    index= int(round(dum[0]))
    plt.imshow(trainX[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: "+names[testY[i]])
    plt.title("Prediction: "+names[index])
    plt.show()