# Keras
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = (128, 128)  # Resize frames to 128x128
NUM_FRAMES = 16        # Number of frames per video
MODEL_PATH = r"path/of/the/saved/deepfake_model.keras"  # Path to the saved model


# Load the trained model
model = load_model(MODEL_PATH)

def preprocess_video(video_path):
    """
    Preprocess the video: extract frames, resize, and prepare them for the model.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    # Read frames from the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= NUM_FRAMES:
            break
        
        # Resize the frame to the target image size
        frame = cv2.resize(frame, IMG_SIZE)
        frames.append(frame)
        frame_count += 1

    cap.release()

    # Ensure there are exactly NUM_FRAMES
    if len(frames) < NUM_FRAMES:
        print("Not enough frames extracted. The video is too short!")
        return None

    # Convert the frames to a numpy array and expand dimensions for batch processing
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    return frames

def predict_deepfake(video_path):
    """
    Predict whether the video is a deepfake or real.
    """
    # Preprocess the video and prepare it for prediction
    frames = preprocess_video(video_path)
    if frames is None:
        return "Error: Video is too short or could not be processed."

    # Make prediction
    prediction = model.predict(frames)
    
    # Interpretation of the prediction
    if prediction[0] > 0.5:
        return "Deepfake Video"
    else:
        return "Real Video"

# Test the model with a sample video
video_path = r"path/of/the/video"  # Replace with the path to the video you want to test
# video_path = r"C:\Users\johnw\Downloads\fake.mp4"  # Replace with the path to the video you want to test
result = predict_deepfake(video_path)
print(f"The video is: {result}")
