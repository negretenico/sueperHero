import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random


#this directory is user specific 
DATADIR = "C:/Users/.../sueprHero"
categories = ['DC','Marvel']

for cat in categories:
    #path for the folder
    path = os.path.join(DATADIR,cat)
    for images in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,images),cv2.IMREAD_GRAYSCALE)
        IMG_SIZE = 150
        newImg = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        plt.imshow(newImg,cmap= "gray")
        plt.show()
        break
    break



training_data  = []

def create_training_data():
    for cat in categories:
        # path for the folder
        path = os.path.join(DATADIR, cat)
        classNum = categories.index(cat)
        for images in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
                IMG_SIZE = 150
                newImg = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([newImg,classNum])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

print(training_data)
trainX =[]
testY =[]

for img, label in training_data:
    trainX.append(img)
    testY.append(label)

trainX = np.array(trainX).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open("trainX.pickle","wb")

pickle.dump(trainX,pickle_out)
pickle_out.close()

pickle_outY = open("testY.pickle","wb")

pickle.dump(testY,pickle_outY)

pickle_out.close()
