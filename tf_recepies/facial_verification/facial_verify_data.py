# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid

def create_directories():
    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')

    # Make the directories
    
    # os.makedirs(POS_PATH)
    # os.makedirs(NEG_PATH)
    # os.makedirs(ANC_PATH)
    print("Created directories")
    return POS_PATH, NEG_PATH,ANC_PATH

#create positive samples images


# os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

def collect_images(POS_PATH, NEG_PATH,ANC_PATH):
    cap = cv2.VideoCapture(1)
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250, :]
        
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path 
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)
        
        # Show image back to screen
        cv2.imshow('Image Collection', frame)
        
        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    POS_PATH, NEG_PATH,ANC_PATH=create_directories()
    collect_images(POS_PATH, NEG_PATH,ANC_PATH)