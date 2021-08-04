from tensorflow import keras
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from data import Data
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip,RandomZoom,RandomRotation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

data = Data()
IMG_HEIGHT = data.img_height
IMG_WIDTH = data.img_width
train_ds = data.train_ds
val_ds = data.val_ds
learning_rate = 1.00e-3

class_names = data.class_names
data_augmentation = Sequential(
  [
    RandomFlip("horizontal_and_vertical",  input_shape=(  IMG_HEIGHT,
                                                                IMG_WIDTH,1)),
Rescaling(scale=1.0 / 255),

RandomZoom(
height_factor=(-0.05, -0.15),
width_factor=(-0.05, -0.15)),
RandomRotation(0.3)]
  
)
batch_size =1
path = 'saved_model/my_model'


categories =  ["Marvel","DC"]
DIR= os.getcwd()+"\\Testing"

if(os.path.isdir(os.path.join(os.getcwd(),path))):
    model = tf.keras.models.load_model(path)
    # Check its architecture
    model.summary()  
  

else:
    model = Sequential([
      data_augmentation,
      Conv2D(16, 3, padding='same', activation='relu'),
      Conv2D(16, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),

      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),

      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(len(class_names))
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])   
    model.summary()
    with tf.device('/GPU:0'):
        epochs = 200
        history = model.fit( train_ds,  validation_data=val_ds,  epochs=epochs)
        model.save('saved_model/my_model')



total = 0
num_correct = 0
["DC","MARVEL"]
for cat in categories:
    for file in os.listdir(os.path.join(DIR,cat)):
        total +=1
        img = tf.keras.preprocessing.image.load_img(os.path.join(os.path.join(DIR,cat),file), color_mode="grayscale",target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        # img = cv2.imread(os.path.join(os.path.join(DIR,cat),file)) 
        
        # # Output img with window name as 'image'
        # cv2.imshow('image', img) 
        
        # # Maintain output window utill
        # # user presses a key
        # cv2.waitKey(0)        
        
        # # Destroying present windows on screen
        # cv2.destroyAllWindows() 
        score = tf.nn.softmax(predictions[0])
        print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a { 100 * np.max(score)} percent confidence.\n")
        print(f"The true label is {cat}\n")
        if cat == class_names[np.argmax(score)]:
          num_correct +=1

print(f"Total Correct:{num_correct}\nTotal Tests:{total}\nAccuracy: {num_correct/total}")
