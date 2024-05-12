# STAGE IV: ACTIVE LEARNING

# I Importing necessary modules and libraries

import collections
import cv2
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scikitplot as skplt
import sys
import tensorflow as tf
from random import shuffle
from imutils import paths
from scipy import stats as st
from sklearn.metrics import confusion_matrix
from sys import argv
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import *


# II. Basic parameters (can be changed)

#script, subdir = argv # Importing information from optimization command file line
working_dir = os.getcwd() # Working directory
file_group_length = 20  # The length of the list of videofiles features extracted together within one cycle (within a single numpy array)
files_dir = 'whole' # Directory containing videofiles (must be in the working directory)
seq_length = 30 # Videofile length (frames number)
num_features = 2048 # The number of features extracted by CNN
batch_size = 64 # Batch size
model_number = 5 # The number of models in ensemble
epochs = 100 # The largest number of training epochs
dir_ending = '' # Can be specified for some tasks
dir_united = 'whole'+dir_ending
cat_dct = {'target_categories': ['A','AC','F','FM','FS','L','P','R','V'],
           'target_categories_category': ['Activity','Copulation attempt','Orientation head-head','Following','Orientation head-abdomen','Licking', 'Prining','Rest','Vibration'], 
'target_categories_binary': ['Non-courtship', 'Coutrship']} # Dictionary containing labels and classes names
write_dir = 'Active_learning' # Directory containing data for active learning
active_model_number = 2  # Model number for active learning
act_multiplier = 0.15 # Parameters that specifies which part of the second set to add to training set after each stage of active learning 

dir_united = 'whole'+dir_ending

cat_dct = {'target_categories': ['A','AC','F','FM','FS','L','P','R','V'],
           'target_categories_category': ['Активность','Попытка копуляции','Ориентация голова к голове','Преследование','Ориентация голова к брюшку','Лизание', 'Прининг','Покой','Вибрация'], 
'target_categories_binary': ['Неухаживание', 'Ухаживание']}

os.chdir(working_dir)

# Prepairs files containing pahts for train and test videos
df = pd.read_csv(working_dir+'/'+dir_united+'/'+dir_united+'.csv', sep = ',')
print(f"Total videos for training: {len(df)}")

# III. Functions

# Functions prepairing data for training

def data_load(read_dir):
    '''Loads data during each stage of active learning'''
    train_data = np.load(read_dir+'/'+'train_data.npy')
    train_mask = np.load(read_dir+'/'+'train_mask.npy')
    train_labels = np.load(read_dir+'/'+'train_labels.npy')
    val_data = np.load(read_dir+'/'+'val_data.npy')
    val_mask = np.load(read_dir+'/'+'val_mask.npy')
    val_labels = np.load(read_dir+'/'+'val_labels.npy')
    test_data = np.load(read_dir+'/'+'test_data.npy')
    test_mask = np.load(read_dir+'/'+'test_mask.npy')
    test_labels = np.load(read_dir+'/'+'test_labels.npy')
    return train_data, train_mask, train_labels, val_data, val_mask, val_labels, test_data, test_mask, test_labels

def data_save(data, mask, labels, val_data, val_mask, val_labels, test_data, test_mask, test_labels, writedir):
    '''Saves updated samplings after active learning stage'''
    with open(writedir+'/'+'train_data.npy', 'wb') as f1:
        np.save(f1, data)
    with open(writedir+'/'+'train_mask.npy', 'wb') as f1:
        np.save(f1, mask)
    with open(writedir+'/'+'train_labels.npy', 'wb') as f1:
        np.save(f1, labels)
    with open(writedir+'/'+'val_data.npy', 'wb') as f1:
        np.save(f1, val_data)
    with open(writedir+'/'+'val_mask.npy', 'wb') as f1:
        np.save(f1, val_mask)
    with open(writedir+'/'+'vaL_labels.npy', 'wb') as f1:
        np.save(f1, val_labels)
    with open(writedir+'/'+'test_data.npy', 'wb') as f1:
        np.save(f1, test_data)
    with open(writedir+'/'+'test_mask.npy', 'wb') as f1:
        np.save(f1, test_mask)
    with open(writedir+'/'+'test_labels.npy', 'wb') as f1:
        np.save(f1, test_labels)

def data2_load(read_dir):
    '''Loads updated parts of additional sampling when performing control leaning for active learning'''
    train_data = np.load(read_dir+'/'+'data_part2.npy')
    train_mask = np.load(read_dir+'/'+'mask_part2.npy')
    train_labels = np.load(read_dir+'/'+'labels_part2.npy')
    return train_data, train_mask, train_labels


def data2_save(data, mask, labels, writedir):
    '''Saves updated parts of additional sampling when performing control leaning for active learning'''
    with open(writedir+'/'+'data_part2.npy', 'wb') as f1:
        np.save(f1, data)
    with open(writedir+'/'+'mask_part2.npy', 'wb') as f1:
        np.save(f1, mask)
    with open(writedir+'/'+'labels_part2.npy', 'wb') as f1:
        np.save(f1, labels)

def categories_save(data, mask, labels, writedir):
    '''Saves arrays for different classes'''
    with open(writedir+'/'+'data.npy', 'wb') as f1:
        np.save(f1, data)
    with open(writedir+'/'+'mask.npy', 'wb') as f1:
        np.save(f1, mask)
    with open(writedir+'/'+'labels.npy', 'wb') as f1:
        np.save(f1, labels)

def category_array_write(category_arr, write_dir):
    '''Writes classes array dictionary to category array directory'''
    for i in range(len(category_arr)):
        subdir = write_dir+'/'+str(i)
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        data = category_arr[i]['data']
        mask = category_arr[i]['mask']
        labels = category_arr[i]['labels']
        categories_save(data=data, mask=mask, labels=labels, writedir = subdir)

def category_array_read(write_dir):
    '''Reads classes directory in write_dir and creates category array dictionary'''
    category_array = {} 
    for i in range(9):
        read_dir = write_dir+'/'+str(i)
        category_array[i] = {'data':[],'mask':[],'labels':[]}
        category_array[i]['data'] = np.load(read_dir+'/'+'data.npy')
        category_array[i]['mask'] = np.load(read_dir+'/'+'mask.npy')
        category_array[i]['labels'] = np.load(read_dir+'/'+'labels.npy')
    return category_array

def dataset_split(dataset_data, dataset_mask, dataset_labels, percent = 0.5, shuffling = True):
    '''Splits dataset to two equal parts'''
    length = len(dataset_data)
    inds = [i for i in range(len(dataset_data))]
    if shuffling == True:
        shuffle(inds)
    full_length = len(inds)
    part_splitter = int(full_length*percent)  
    part1_inds = inds[:part_splitter]
    part1_data = np.take(dataset_data, indices=part1_inds, axis=0)
    part1_mask = np.take(dataset_mask, indices=part1_inds, axis=0)
    part1_labels = np.take(dataset_labels, indices=part1_inds, axis=0)
    part2_inds = inds[part_splitter:]
    part2_data = np.take(dataset_data, indices=part2_inds, axis=0)
    part2_mask = np.take(dataset_mask, indices=part2_inds, axis=0)
    part2_labels = np.take(dataset_labels, indices=part2_inds, axis=0)
    return part1_data, part1_mask, part1_labels, part2_data, part2_mask, part2_labels

def sampling_add(accuracies, train_data, act_multiplier):
    '''Calculates numbers of elements to be added to training set during active learning'''
    false = [1-i for i in accuracies] 
    false = false/sum(false)
    length = len(df)-len(train_data)
    new_sampling_sizes = (length*act_multiplier*false).tolist()
    new_sampling_sizes = [int (i) for i in new_sampling_sizes]
    return new_sampling_sizes

def active_learning_select(data, mask, labels, category_array, new_sampling_sizes):
    '''Adds samples to training sample proportional to the frequency of their errors'''
    for i in range(9):
        length = len(category_array[i]['data'])
        cat_inds = [i for i in range(length)]
        shuffle(cat_inds)
        if length >0:
            if new_sampling_sizes[i] < length:
                inds_sel = cat_inds[:new_sampling_sizes[i]]
            else:
                inds_sel = cat_inds
            data_add = np.take(category_array[i]['data'], indices=inds_sel, axis=0)
            category_array[i]['data'] = np.delete(category_array[i]['data'], inds_sel, axis = 0)
            mask_add = np.take(category_array[i]['mask'], indices=inds_sel, axis=0)
            category_array[i]['mask'] = np.delete(category_array[i]['mask'], inds_sel, axis = 0)
            labels_add = np.take(category_array[i]['labels'], indices=inds_sel, axis=0)
            category_array[i]['labels'] = np.delete(category_array[i]['labels'], inds_sel, axis = 0)
            data = np.concatenate((data, data_add),axis = 0)
            mask = np.concatenate((mask, mask_add),axis = 0)
            labels = np.concatenate((labels, labels_add),axis = 0)
    return data, mask, labels, category_array

def active_learning_select_control(train_data, data_part2, train_mask, mask_part2, train_labels, labels_part2, length_value):
    '''Adds samples to training sample regardless of the frequency of their errors'''
    length = length_value
    new_sampling_sizes = [i for i in range(len(data_part2))]
    inds = new_sampling_sizes[:length]
    shuffle(inds)
    add_data = np.take(data_part2, indices=inds, axis=0)
    data_part2 = np.delete(data_part2, inds, axis = 0)
    add_mask = np.take(mask_part2, indices=inds, axis=0)
    mask_part2 = np.delete(mask_part2, inds, axis = 0)
    add_labels = np.take(labels_part2, indices=inds, axis=0)
    labels_part2 = np.delete(labels_part2, inds, axis = 0)
    train_data = np.concatenate((train_data, add_data),axis = 0)
    train_mask = np.concatenate((train_mask, add_mask),axis = 0)
    train_labels = np.concatenate((train_labels, add_labels),axis = 0)
    data_save(data=train_data, mask=train_mask, labels=train_labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, writedir=write_dir)
    data2_save(data=data_part2, mask=mask_part2, labels=labels_part2, writedir=write_dir)




# Functions for different models

def model_3(write_dir, initial_learning_rate=0.001, load_weights=False, model_number=0, load_final_weights=False, weights_file = "0.hdf5"):
    ''' Model 3 construction'''
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((seq_length, num_features))
    mask_input = keras.Input((seq_length,), dtype="bool")
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences = True, return_state = True, dropout = 0.2, recurrent_dropout = 0.2))(
        frame_features_input, mask=mask_input  
    )
    x = keras.layers.GRU(64, dropout = 0.2, recurrent_dropout = 0.2)(x[0])
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)
    rnn_model = keras.Model([frame_features_input, mask_input], output)
    lr = ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000, decay_rate=0.9)
    if load_final_weights is True:
        rnn_model.load_weights(write_dir+'/'+weights_file)
    else:
        if load_weights is True:
            rnn_model.load_weights(write_dir+'/'+str(model_number)+".hdf5")
        rnn_model.compile(
            loss="sparse_categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(lr), metrics=["accuracy"]
    )
    return rnn_model

# Functions performing model optimization

def model_train_active_learning(train_data, train_mask, train_labels, write_dir, model, model_number):
    ''' Trains n different models during active learning using different train and validation splittings'''
    for model_number in range(model_number):
        data, mask, labels, val_data, val_mask, val_labels = dataset_split(dataset_data=train_data, dataset_mask=train_mask, dataset_labels=train_labels, percent = 0.8, shuffling = True)
        history, sequence_model = run_experiment(initial_learning_rate = 1e-4, load_weights = True, early = True, model_number = model_number, model_basis = model, train_data=data, train_mask=mask, train_labels=labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, write_dir = write_dir)
        history, sequence_model = run_experiment(initial_learning_rate = 2e-5, load_weights = True, early = True, model_number = model_number, model_basis = model, train_data=data, train_mask=mask, train_labels=labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, write_dir = write_dir)
    
    return history, sequence_model

def run_experiment(initial_learning_rate, load_weights, early, model_number, model_basis, train_data, train_mask, train_labels, val_data, val_mask, val_labels, test_data, test_mask, test_labels, write_dir):  # Here the mode of training can be specified
    ''' Performs a single stage of model optimization (for given weights and learning rate'''
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    checkpoint = keras.callbacks.ModelCheckpoint(write_dir+'/'+str(model_number)+'.hdf5' , monitor = ['train_loss'] , verbose = 1  , mode = 'max')
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    if early == True:
        callbacks_list = [checkpoint, earlystop]
    else:
        callbacks_list = [checkpoint]
    seq_model =model_basis(write_dir = write_dir, initial_learning_rate = initial_learning_rate, load_weights = load_weights, model_number=model_number)
    history = seq_model.fit(
        x= [train_data, train_mask], y = train_labels,
        validation_data = ([val_data, val_mask], val_labels), # Model is evaluated each epoch using test dataset.
        validation_split=0.0,
        epochs=epochs,
        callbacks = callbacks_list
    ) # All metrics will be written in history. 
    _, accuracy = seq_model.evaluate([val_data, val_mask], val_labels)  # Final model evaluation.
    print(f"Val accuracy: {round(accuracy * 100, 2)}%")
    return history, seq_model

# Evaluation functions

def predictions_single(write_dir, val, val_labels, current_model, model_number):
    '''Predicts video classes for a single trained model'''
    model = current_model(write_dir=write_dir, initial_learning_rate=0.001, load_weights=False, load_final_weights=True, weights_file = str(sample)+'.hdf5')
    Y_preds_category = model.predict(val).argmax(axis=-1)
    binary_decoder = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:0, 7:0, 8:1}
    Y_preds_binary = []
    j = 0
    for i in Y_preds_category:
        j = binary_decoder[i]
        Y_preds_binary.append(j) 
    Y_labels = val_labels
    Y_labels_category = Y_labels.T
    Y_labels_category = Y_labels_category[0]
    
    Y_labels_binary = []
    j = 0
    for i in Y_labels_category:
        j = binary_decoder[i]
        Y_labels_binary.append(j)
    cat_dct['Y_preds_category'] = Y_preds_category
    cat_dct['Y_preds_binary'] = Y_preds_binary
    cat_dct['Y_labels_category'] = Y_labels_category
    cat_dct['Y_labels_binary'] = Y_labels_binary
    return cat_dct


def predictions_voting(write_dir, val, val_labels, current_model, model_number):
    ''' Predicts video classes based on several separately trained models'''
    models_array = []
    for sample in range(model_number):
        model = current_model(write_dir=write_dir, initial_learning_rate=0.001, load_weights=False, load_final_weights=True, weights_file = str(sample)+'.hdf5')
        Y_preds_category = model.predict(val).argmax(axis=-1)
        models_array.append(Y_preds_category)
    predictions_array = np.array(models_array)
    Y_preds_category = list(st.mode(predictions_array)[0])

    binary_decoder = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:0, 7:0, 8:1}

    Y_preds_binary = []
    j = 0
    for i in Y_preds_category:
        j = binary_decoder[i]
        Y_preds_binary.append(j)
        
    Y_labels = val_labels
    Y_labels_category = Y_labels.T
    Y_labels_category = Y_labels_category[0]
    
    Y_labels_binary = []
    j = 0
    for i in Y_labels_category:
        j = binary_decoder[i]
        Y_labels_binary.append(j)
    cat_dct['Y_preds_category'] = Y_preds_category
    cat_dct['Y_preds_binary'] = Y_preds_binary
    cat_dct['Y_labels_category'] = Y_labels_category
    cat_dct['Y_labels_binary'] = Y_labels_binary
    return cat_dct

def accuracy_classes(cat_dct, category_type='category', for_writing = True):
    '''Calculates model accuracies'''
    from sklearn.metrics import confusion_matrix
    Y_true = cat_dct['Y_labels_'+category_type]
    Y_pred = cat_dct['Y_preds_'+category_type]
    matrix = confusion_matrix(Y_true, Y_pred)
    accuracies = np.round(matrix.diagonal()/matrix.sum(axis=1),2)
    if for_writing == True:
        acc = str(accuracies)
        acc = acc.replace('  ', ' ')
        acc = acc.replace(' ',',')
        acc = acc.replace('[','')
        accuracies = acc.replace(']','')
    return accuracies

# IV. Control learning

train_data, train_mask, train_labels, val_data, val_mask, val_labels, test_data, test_mask, test_labels=data_load(write_dir) # Load samplings for active learning
data_part2, mask_part2, labels_part2 = data2_load(write_dir)

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["tag"]))
print(label_processor.get_vocabulary())

print (f': data_shape: {train_data.shape} data_mask_shape: {train_mask.shape} data_labels_shape: {train_labels.shape} val_data_shape: {val_data.shape} val_mask_shape: {val_mask.shape} val_labels_shape: {val_labels.shape} test_data_shape: {test_data.shape} test_mask_shape: {test_mask.shape} test_labels_shape: {test_labels.shape}')
history, model = model_train_active_learning(train_data = train_data, train_mask = train_mask, train_labels = train_labels, write_dir = write_dir, model = model_3, model_number = active_model_number)

for sample in range(model_number):
    cat_dct = predictions_single(write_dir=write_dir, val=[test_data, test_mask], val_labels=test_labels, current_model=model_3, model_number=active_model_number)
    accuracies = accuracy_classes(cat_dct=cat_dct, category_type='category')
    print(f'Sample: {sample}  Accuracies: {accuracies} Dataset length: {len(train_data)}')
    with open (r'Active_learning_new.csv','a') as f:
        f.write(f'{sample}, {accuracies},{len(train_data)}\n')

cat_dct = predictions_voting(write_dir = write_dir, val=[test_data, test_mask], val_labels=test_labels, current_model=model_3, model_number = active_model_number)
accuracies = accuracy_classes(cat_dct, category_type='category')
print(f'Voting_accuracies : {accuracies} Dataset length: {len(train_data)}\n')
with open (r'Active_learning_new.csv','a') as f:
    f.write(f'Voting_accuracies, {accuracies},{len(train_data)}\n')

accuracies = accuracy_classes(cat_dct, category_type='category', for_writing = False)
active_learning_select_control(train_data=train_data, data_part2=data_part2, train_mask=train_mask, mask_part2=maks_part2, train_labels=train_labels, labels_part2=labels_part2, length_value=length_value)
