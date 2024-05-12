# STAGE II: VIDEO PREPROCESSING AND FEATURES EXTRACTION USING PRETRAINED CONVOLUTIONARY NEURAL NETWORK (CNN)

# I Importing necessary modules and libraries

import cv2
import gc
import glob
import numpy as np
import pandas as pd
import os
import sys
from imutils import paths
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from sys import argv
from tensorflow import keras

# II. Basic parameters (can be changed)

script, videofiles_dir, start, file_group_length, cnn_type, processing_type = argv # Importing information from features_extracting command file line
image_size_heigth = 270 # Videofile frame heigth
image_size_width = 270 # Videofile frame width
seq_length = 30 # Videofile length (frames number)
num_features = 2048 # The number of features extracted by CNN
dir_ending = '' # Can be specified for some tasks
working_dir = os.getcwd() # Working directory

# III. FUNCTIONS

def image_process(img, k =1.2):
    ''' Images processing: invertes colors and removes background'''
    new_image = (255-img)
    threshold = np.mean(new_image)*k
    x = threshold
    new_image[np.all(new_image <= (x, x, x), axis=-1)] = (0,0,0)
    return new_image

def load_video(path, max_frames=0, resize=(image_size_heigth, image_size_width), processing_type=processing_type):
    '''Loads resizes and processes videos'''
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if processing_type == 'inverted':
                frame = image_process(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    print (np.array(frames).shape)
    return np.array(frames)

def build_feature_extractor(cnn_type):
    '''Configurates neural network for videos features extraction'''
    inputs = keras.Input((image_size_heigth, image_size_width, 3))
    if cnn_type == 'EfficientNet_B5':
        feature_extractor = keras.applications.efficientnet.EfficientNetB5(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_size_heigth, image_size_width, 3),)
        preprocess_input = keras.applications.efficientnet.preprocess_input
    elif cnn_type == 'Inception_V3':
        feature_extractor = keras.applications.InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_size_heigth, image_size_width, 3),)
        preprocess_input = keras.applications.inception_v3.preprocess_input
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def prepare_videos(df):
    '''Prepares data including videos paths and labels'''
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    return video_paths, labels

def prepare_all_videos(video_paths_part, root_dir, feature_extractor):
    '''Preprocessed videofiles: features extraction'''
    num_samples = len(video_paths_part)
    frame_masks = np.zeros(shape=(num_samples, seq_length), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, seq_length, num_features), dtype="float32"
    )
    # For each video.
    for idx, path in enumerate(video_paths_part):
        # Gathers all its frames and adds a batch dimension.
        path_local = os.path.join(root_dir, path)
        print(path_local)
        frames = load_video(path = path_local, processing_type=processing_type)
        frames = frames[None, ...]

        # Initializes placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, seq_length,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, seq_length, num_features), dtype="float32"
        )

        # Extracts features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(seq_length, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    arr =  (frame_features, frame_masks)
    return arr

def write_video_features_tensor(start, values_part, labels_part, writedir):
    '''Writes video preprocessed data in numpy array form'''
    with open(writedir+'/'+str(start)+'_labels.npy', 'wb') as f1:
        np.save(f1, labels_part) # Array containing labels
    x_part = values_part[0]
    with open(writedir+'/'+str(start)+'_data.npy', 'wb') as f1:
        np.save(f1, x_part) # Array containing features data extracted by CNN
    x_part = values_part[1]
    with open(writedir+'/'+str(start)+'_data_mask.npy', 'wb') as f1:
        np.save(f1, x_part) # Array containing masks (whether the given frame will be processed or not)

def processing_videos(videofiles_dir, start, file_group_length, cnn_type):
    '''Performs video features extraction and writes numpy arrays'''
    root_dir = videofiles_dir_work
    writedir = videofiles_dir_work+'_features'
    if not os.path.exists(writedir):
        os.mkdir(writedir) # Creates folder with numpy arrays containing extracted features. !!! Must be updated for each new features calculation
    video_paths, labels = prepare_videos(df=df)
    num_samples = len(video_paths)
    feature_extractor = build_feature_extractor(cnn_type=cnn_type)
    video_paths_part = video_paths[start*file_group_length: min((start+1)*file_group_length, num_samples)]
    labels_part = labels[start*file_group_length: min((start+1)*file_group_length, num_samples)]
    values_part = prepare_all_videos(video_paths_part=video_paths_part, root_dir = root_dir, feature_extractor = feature_extractor)
    write_video_features_tensor(start=start, values_part=values_part, labels_part=labels_part, writedir=writedir)

# IV. Features extraction

os.chdir(working_dir)
videofiles_dir_work = videofiles_dir+dir_ending # Directory containing videofiles
df = pd.read_csv(videofiles_dir_work+'/'+videofiles_dir_work+'.csv', sep = ',') # Dataframe containing information about videofiles

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df["tag"]) # Creates vocabulary containing videofiles labels
)
print(label_processor.get_vocabulary()) # Prints vocabulary containing videofiles labels

processing_videos(videofiles_dir = videofiles_dir, start = int(start), file_group_length = int(file_group_length), cnn_type=cnn_type) # Processing videos and writing numpy arrays

gc.collect()
sys.modules[__name__].__dict__.clear()
