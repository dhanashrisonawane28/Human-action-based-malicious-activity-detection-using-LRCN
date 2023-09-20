import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import *

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def visulizedata():
    # Create a Matplotlib figure and specify the size of the figure.
    print("Inside")
    plt.figure(figsize=(20, 20))

    # Get the names of all classes/categories in UCF50.
    all_classes_names = os.listdir('C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet')

    # Generate a list of 20 random values. The values will be between 0-50,
    # where 50 is the total number of class in the dataset.
    random_range = random.sample(range(len(all_classes_names)), 5)

    # Iterating through all the generated random values.
    for counter, random_index in enumerate(random_range, 1):
        # Retrieve a Class Name using the Random Index.
        selected_class_Name = all_classes_names[random_index]

        # Retrieve the list of all the video files present in the randomly selected Class Directory.
        video_files_names_list = os.listdir(
            f'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/{selected_class_Name}')

        # Randomly select a video file from the list retrieved from the randomly selected Class Directory.
        selected_video_file_name = random.choice(video_files_names_list)

        # Initialize a VideoCapture object to read from the video File.
        video_reader = cv2.VideoCapture(
            f'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/{selected_class_Name}/{selected_video_file_name}')

        # Read the first frame of the video file.
        _, bgr_frame = video_reader.read()

        # Release the VideoCapture object.
        video_reader.release()

        # Convert the frame from BGR into RGB format.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Write the class name on the video frame.
        cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame.
        plt.subplot(5, 4, counter);
        plt.imshow(rgb_frame);
        plt.axis('off')

        #displat plot
        plt.show()


def main():
    visulizedata()


if __name__ == "__main__":
    main()
