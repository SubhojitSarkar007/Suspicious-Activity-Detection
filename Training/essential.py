import numpy as np
import os
import cv2

import matplotlib
import matplotlib.pyplot as plt
import math

import keras
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, AveragePooling3D,
                          Reshape, Lambda, GlobalAveragePooling3D, Concatenate,
                          ReLU, Add)

from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from keras.backend import set_session

from keras.callbacks import CSVLogger

## ESSENTIAL CUSTOM FUNCTIONS

#PLOT FUNCTION
def plot_history(history, result_dir):
    '''
    Plots the accuracy and loss graphs of train and val and saves them.
    '''

    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.show()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.show();
    
# VIDEO TO 3D

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename):
        
        frames = []
        index = len(os.listdir(filename)) // self.depth
        images = os.listdir(filename)[::index]
        images = images[0:25]
        images.sort()

        for img in images:

            img_path = os.path.join(filename, img)
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (self.height, self.width))
            frames.append(frame)

        return np.array(frames) / 255.0
    
    
def preprocess(video_dir, result_dir, nb_classes = 101, img_size = 224, frames = 25):
    '''
    Preprocess the videos into X and Y and saves in npz format and 
    computes input shape
    '''

    img_rows, img_cols  = img_size, img_size

    channel = 3

    files = os.listdir(video_dir)
    files.sort()

    if '.ipynb_checkpoints' in files:
        files.remove('.ipynb_checkpoints')

    X = []
    labels = []
    labellist = []

    # Obtain labels and X
    for filename in files:

        name = os.path.join(video_dir, filename)
        
        for v_files in os.listdir(name):
            
            v_file_path = os.path.join(name, v_files)
            label = filename
            if label not in labellist:
                if len(labellist) >= nb_classes:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(v_file_path)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{} {}\n'.format(i, labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
                
    Y = np_utils.to_categorical(labels, nb_classes)

    print('X_shape:{}\tY_shape:{}'.format(len(X), Y.shape))

    input_shape = (frames, img_rows, img_cols, channel)

    return X, Y, input_shape

    
class batchGenerator(keras.utils.all_utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, vid3d):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.vid3d = vid3d

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []

        for video in self.x[idx * self.batch_size:(idx + 1) * self.batch_size]:
            batch_x.append(self.vid3d.video3d(video))

        batch_x = np.array(batch_x)
        batch_x = batch_x.astype('float32')
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

    
#VIDEO TO FRAME CONVERSION

def frames_from_video(video_dir, nb_frames = 25, img_size = 224):

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0

# PREDICTION FUNCTIONS
def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])

    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))
        return(preds[i])
	

      

