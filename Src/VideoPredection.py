# Import the required libraries.
import os

import PIL
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
from Telegram import telegram_send
from pyglet import media
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import *
from keras.layers import *
from keras import *
from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.models import load_model
from tqdm.asyncio import tqdm

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Assault_Fighting","Explosion","RoadAccidents","Regular_Activties"]

LRCN_model = tf.keras.models.load_model('C:/Users/Vishal/PycharmProjects/AnamolyDetection/Models/Final_3_90_regularadded_ours_actual.h5')


def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Get the probability of the predicted class.
            predicted_class_probability = predicted_labels_probabilities[predicted_label]

            # Display predicted class name along with probability on the frame.
            text = f'{predicted_class_name} ({predicted_class_probability:.2f})'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if the probability is above 88%.
            if predicted_class_probability > 0.88:

                # Display the text for 3 seconds.
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            1000 // 3)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()

def predict_single_action(video_file_path, SEQUENCE_LENGTH):

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    highest_probability = 0
    highest_probability_frame_index = 0
    if predicted_labels_probabilities.max() > highest_probability:
        highest_probability = predicted_labels_probabilities.max()
        highest_probability_frame_index = frame_counter

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Set the video to the frame with the highest predicted probability.
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, highest_probability_frame_index * skip_frames_window)
    _, selected_frame = video_reader.read()

    # Save the selected frame as an image.
    selected_frame_path = "selected_frame.jpg"
    cv2.imwrite(selected_frame_path, selected_frame)

    # Display the information.
    print(f'Highest Predicted Probability: {highest_probability}')
    print(f'Selected Frame Index: {highest_probability_frame_index}')
    print(f'Selected Frame Saved as: {selected_frame_path}')


    # Release the VideoCapture object.
    video_reader.release()

    return predicted_class_name,predicted_labels_probabilities[predicted_label]



def main():

    # Construct the output video path.
    Input_File='C:/Users/Vishal/PycharmProjects/AnamolyDetection/InputVideos/Assault004_x264_segment_2.mp4'
    output_video_file_path = f'C:/Users/Vishal/PycharmProjects/AnamolyDetection/Results/Result_explosion.mp4'

    # # Perform Action Recognition on the Test Video.
    # predict_on_video(Input_File, output_video_file_path, SEQUENCE_LENGTH)
    # # # # # Display the output video.
    # VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))

    # vartoPass1,vartoPass2=predict_single_action(Input_File, SEQUENCE_LENGTH)
    # VideoFileClip(Input_File, audio=False, target_resolution=(300, None))
    # if vartoPass1 != 'Regular_Activties':
    #   telegram_send(vartoPass1,vartoPass2);

    # Initialize the VideoCapture object to capture video from the camera.
    camera = cv2.VideoCapture(0)  # 0 corresponds to the default camera
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    confidence_percentage = 0
    while True:
        # Read the frame from the camera.
        ok, frame = camera.read()

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Append the pre-processed frame into the frames queue.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of the class with the highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            confidence_percentage = predicted_labels_probabilities[predicted_label] * 100
        # Write the predicted class name on top of the frame.
        text = f"{predicted_class_name} ({confidence_percentage:.2f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the predicted action.
        cv2.imshow("Live Action Recognition", frame)

        # Break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the OpenCV windows.
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()













