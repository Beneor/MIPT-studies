
# STAGE I: PREPARING FOR VIDEO FEATURES EXTRACTING

# I Importing necessary modules and libraries

import cv2
import glob
import os
import pandas as pd
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# II. Basic parameters (can be changed)

working_dir = os.getcwd() # Working directory
file_group_length = 20 # The length of the list of videofiles that will be processed together within one cycle (one line in a features_extracting command file)
video_extension = 'mp4' # Videofiles extension 
courtship_lst = ['A', 'AC', 'F', 'FM', 'FS', 'P', 'R', 'V', 'L'] # List of courtship elements
cnn_type = 'EfficientNet_B5' # Type of pretrained convolutional network.
processing_type = 'none' # Type of videofiles processing.
files_dir = 'whole' # Directory containing videofiles (must be in the working directory)

# III. Functions

def change_lst(lst, files_dir):
    ''' Creates csv file containing paths to videofiles and their labels'''
    current_dir = os.path.join(working_dir, files_dir)
    os.chdir(current_dir)
    filesMask='*.'+ video_extension 
    resultFiles = glob.glob(filesMask)
    with open(current_dir+'/'+files_dir+'.csv', 'w') as f:
        caption = 'video_name,tag'
        f.write(caption)
        f.write('\n')
        for filename in resultFiles:
            category = filename[:filename.find('-')]
            if category in lst:
                str = filename+','+category
                f.write(str)
                f.write('\n')

# IV. Prepairing csv file containing paths to videofiles and their labels

change_lst(courtship_lst, files_dir=files_dir)
os.chdir(working_dir)
path=files_dir+'/whole.csv'
df = pd.read_csv(os.path.join(working_dir, path), sep = ',') # Dataframe containing information about videofiles

print(f'Total videos for training: {len(df)}')

# V Prepairing command file for feature extraction using pretrained convolutional network

counts = len(df)//file_group_length 
residue = len(df)%file_group_length
if residue !=0:
    counts +=1

with open('features_extracting.bat', 'w') as f:  # Creates a command file containing paremeters necessary to exctract features from videofiles using pretrained convolutional network
    for count_ind in range(counts):
        f.write('python features_extraction_training.py'+' '+files_dir+' '+str(count_ind)+' '+str(file_group_length)+' '+cnn_type+' '+processing_type)
        f.write('\n')

sys.modules[__name__].__dict__.clear()

