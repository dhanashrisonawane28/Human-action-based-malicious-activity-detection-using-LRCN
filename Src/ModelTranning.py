# Import the required libraries.
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed

from moviepy.editor import *


from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import *
from keras.layers import *
from keras import *
from keras.utils.vis_utils import plot_model

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 60
# Specify the directory containing the UCF50 dataset.
DATASET_DIR = "C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/Newfolder"
# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Assault","RoadAccidents"]

train_dir = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/Train'
test_dir = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/Test'

features_train = np.load(os.path.join(train_dir, 'features_train.npy'))
labels_train = np.load(os.path.join(train_dir, 'labels_train.npy'))
features_test = np.load(os.path.join(test_dir, 'features_test.npy'))
labels_test = np.load(os.path.join(test_dir, 'labels_test.npy'))

def create_LRCN_model():
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(32))

    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    ########################################################################################################################

    # Display the models summary.
    model.summary()

    # Return the constructed LRCN model.
    return model

def plot_metric(history, metric_name, val_metric_name, title):
    epochs = range(1, len(history.history[metric_name]) + 1)
    plt.plot(epochs, history.history[metric_name], 'bo', label=f'Training {metric_name}')
    plt.plot(epochs, history.history[val_metric_name], 'b', label=f'Validation {val_metric_name}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

def plot_metric(history, metric_name, val_metric_name, title):
    epochs = range(1, len(history.history[metric_name]) + 1)
    plt.plot(epochs, history.history[metric_name], 'bo', label=f'Training {metric_name}')
    plt.plot(epochs, history.history[val_metric_name], 'b', label=f'Validation {val_metric_name}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

def main():
    # Construct the required LRCN model.
    LRCN_model = create_LRCN_model()

    # Display the success message.
    print("Model Created Successfully!")

    plot_model(LRCN_model, to_file='../Images/LRCN_model_structure_plot.png', show_shapes=True, show_layer_names=True)

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    # Compile the model and specify loss function, optimizer and metrics to the model.
    LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # Start training the model.
    LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=30, batch_size=3,
                                                 shuffle=True, validation_split=0.2,
                                                 callbacks=[early_stopping_callback])

    model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)

    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    # Define the string date format.
    # Get the current Date and Time in a DateTime Object.
    # Convert the DateTime object to string according to the style mentioned in date_time_format string.
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

    # Save the Model.
    LRCN_model.save(model_file_name)

    plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')


if __name__ == "__main__":
    main()