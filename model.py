import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten, LSTM, Attention, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATA_PATH = r"path/of/the/datas"
BATCH_SIZE = 8  # Increased batch size
EPOCHS = 10  # Increased number of epochs
IMG_SIZE = (128, 128)  # Resize frames to 128x128
NUM_FRAMES = 16  # Number of frames per video

# Data Generator
def data_generator(data_path, category, batch_size=8):
    
    fake_dir = os.path.join(data_path, "FAKE", category)
    real_dir = os.path.join(data_path, "REAL", category)

    fake_videos = [os.path.join(fake_dir, folder) for folder in os.listdir(fake_dir)]
    real_videos = [os.path.join(real_dir, folder) for folder in os.listdir(real_dir)]

    all_videos = [(video, 0) for video in fake_videos] + [(video, 1) for video in real_videos]  # 0 for FAKE, 1 for REAL

    while True:
        np.random.shuffle(all_videos)  # Shuffle videos for randomness
        batch_frames, batch_labels = [], []

        for video_path, label in all_videos:
            frames = []
            try:
                # Collect NUM_FRAMES frames from the video folder
                frame_files = sorted(os.listdir(video_path))[:NUM_FRAMES]
                for frame_file in frame_files:
                    frame_path = os.path.join(video_path, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.resize(frame, IMG_SIZE)
                        frames.append(frame)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

            if len(frames) == NUM_FRAMES:  # Ensure batch is complete
                batch_frames.append(np.array(frames))
                batch_labels.append(label)

            if len(batch_frames) == batch_size:
                yield np.array(batch_frames), np.array(batch_labels)
                batch_frames, batch_labels = [], []
 
#
def hybrid_model(input_shape):

    # Pre-trained EfficientNet backbone (can replace with Swin Transformer if available)
    swin_transformer = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(input_shape[1], input_shape[2], input_shape[3]), weights='imagenet'
    )
    swin_transformer.trainable = False  # Freeze pre-trained weights

    model_input = layers.Input(shape=input_shape)
    
    # TimeDistributed Swin Transformer for spatial features extraction
    time_distributed = layers.TimeDistributed(swin_transformer)(model_input)
    time_distributed = layers.TimeDistributed(Flatten())(time_distributed)
    
    # LSTM for temporal feature extraction
    lstm = LSTM(256, return_sequences=True)(time_distributed)  # Increased LSTM units
    attention = Attention()([lstm, lstm])
    
    # Refined LSTM after attention
    refined_lstm = LSTM(128, return_sequences=False)(attention)
    
    # Dense layers for classification
    dense = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(refined_lstm)  # L2 regularization
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(model_input, output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model input shape
input_shape = (NUM_FRAMES, *IMG_SIZE, 3)  # Sequence of 16 frames, each 128x128 with 3 channels

# Create the model
model = hybrid_model(input_shape)
model.summary()

# Data Generators
train_gen = data_generator(DATA_PATH, "TRAIN", BATCH_SIZE)
val_gen = data_generator(DATA_PATH, "VAL", BATCH_SIZE)

# Callbacks for learning rate reduction and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model
model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=10,
    validation_steps=5,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping]
)

# Save Model
model.save("deepfake_model.keras")
print("Model saved successfully!")
