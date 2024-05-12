# STAGE III: MODEL TRAINING AND ACCURACY CHECK

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
subdir = '1'


# III. Functions

# Functions prepairing data for training

def arr_concatenation(files_dir, dir_ending, suffix, processing_suffix):
    ''' Concatenates numpy arrays for video files'''
    path = files_dir+dir_ending+processing_suffix
    current_dir = os.path.join(working_dir, path)
    os.chdir(current_dir)
    suffix = suffix
    sum_file = np.load('0_'+suffix+'.npy')
    newsuffix = suffix+'.npy'
    print(newsuffix)
    dct = {}
    filesMask='*'+newsuffix
    resultFiles = glob.glob(filesMask)
    for filename in resultFiles:
        with open(filename, 'r') as f:
            file_ind = int(filename[:filename.find('_')])
            dct[file_ind] = filename
    sorted_docs = collections.OrderedDict(sorted(dct.items())) # Arrays will be concatenated according to the order of their generation.
    #print(sorted_docs)
    lst = []
    for key in sorted_docs:
        lst.append(sorted_docs[key])
    for i in range(1, len(lst)):
        f1 = np.load(lst[i])
        sum_file = np.concatenate((sum_file,f1),axis = 0)
    return sum_file

def data_preparing(files_dir, processing_suffix='_features'):
    '''Prepares numpy arrays for model training'''
    data_part = arr_concatenation(files_dir=files_dir, dir_ending = dir_ending, suffix='data', processing_suffix = processing_suffix)
    data_mask = arr_concatenation(files_dir=files_dir,  dir_ending = dir_ending, suffix='data_mask', processing_suffix = processing_suffix)
    labels = arr_concatenation(files_dir=files_dir,  dir_ending = dir_ending, suffix='labels', processing_suffix = processing_suffix)
    return data_part, data_mask, labels

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


def category_select(dataset_data, dataset_mask, dataset_labels):
    '''Creates dictionary containing arrays of extracted features for 9 separate classes of videofiles'''
    cat_ind_lst = [[],[],[],[],[],[],[],[],[]]
    category_array = {}
    for i in range(9):
        for j in range(len(dataset_labels)):
            if dataset_labels[j][0] == i:
                cat_ind_lst[i].append(j)
    for i in range(9):
        category_array[i] = {'data':[],'mask':[],'labels':[]}
        category_array[i]['data'] = np.take(dataset_data, indices = cat_ind_lst[i], axis = 0)
        category_array[i]['mask'] = np.take(dataset_mask, indices = cat_ind_lst[i], axis = 0)
        category_array[i]['labels'] = np.take(dataset_labels, indices = cat_ind_lst[i], axis = 0)
    return cat_ind_lst, category_array


def dataset_train_val_shuffle(inds, dataset_data, dataset_mask, dataset_labels, val_shuffling = True):
    '''Splits features arrays to train, validation and test sets'''
    full_length = len(inds)
    test_splitter = int(full_length*0.8)
    
    test_inds = inds[test_splitter:]
    test_data = np.take(dataset_data, indices=test_inds, axis=0)
    test_mask = np.take(dataset_mask, indices=test_inds, axis=0)
    test_labels = np.take(dataset_labels, indices=test_inds, axis=0)
    
    train_val_inds = inds[0:test_splitter]
    train_val_length = len(train_val_inds)
    if val_shuffling == True:
        shuffle(train_val_inds)
    val_splitter = int(train_val_length*0.8)
    train_inds = train_val_inds[0:val_splitter]
    val_inds = train_val_inds[val_splitter:]
    train_data = np.take(dataset_data, indices=train_inds, axis=0)
    train_mask = np.take(dataset_mask, indices=train_inds, axis=0)
    train_labels = np.take(dataset_labels, indices=train_inds, axis=0)
    val_data = np.take(dataset_data, indices=val_inds, axis=0)
    val_mask = np.take(dataset_mask, indices=val_inds, axis=0)
    val_labels = np.take(dataset_labels, indices=val_inds, axis=0)
    return train_data, train_mask,train_labels, val_data,val_mask,val_labels,test_data,test_mask,test_labels

# Functions for different models

def model_0(write_dir, initial_learning_rate=0.001, load_weights=False, model_number=0, load_final_weights=False, weights_file = "0.hdf5"):
    ''' Model 0 construction'''
    class_vocab = label_processor.get_vocabulary()
    frame_features_input = keras.Input((seq_length, num_features))
    mask_input = keras.Input((seq_length,), dtype="bool")
    x = keras.layers.LSTM(32, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
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

def model_1(write_dir, initial_learning_rate=0.001, load_weights=False, model_number=0, load_final_weights=False, weights_file = "0.hdf5"):
    ''' Model 1 construction'''
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((seq_length, num_features))
    mask_input = keras.Input((seq_length,), dtype="bool")
    x = keras.layers.GRU(32, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
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


def model_2(write_dir, initial_learning_rate=0.001, load_weights=False, model_number=0, load_final_weights=False, weights_file = "0.hdf5"):
    ''' Model 2 construction'''
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((seq_length, num_features))
    mask_input = keras.Input((seq_length,), dtype="bool")
 
    x = keras.layers.GRU(64, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
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


def model_4(write_dir, initial_learning_rate=0.001, load_weights=False, model_number=0, load_final_weights=False, weights_file = "0.hdf5"):
    ''' Model 4 construction'''
    class_vocab = label_processor.get_vocabulary()
    frame_features_input = keras.Input((seq_length, num_features))
    mask_input = keras.Input((seq_length,), dtype="bool")
    x = keras.layers.GRU(64, return_sequences = True, return_state = True, dropout = 0.2, recurrent_dropout = 0.2)(
        frame_features_input, mask=mask_input  
    )
    x = keras.layers.GRU(64, return_sequences = True, return_state = True, dropout = 0.2, recurrent_dropout = 0.2)(x)
    x = keras.layers.GRU(64, return_sequences = True, return_state = True, dropout = 0.2, recurrent_dropout = 0.2)(x)
    x = keras.layers.GRU(64, return_sequences = True, return_state = False, dropout = 0.2, recurrent_dropout = 0.2)(x)
    x = keras.layers.GRU(64, return_sequences = False, return_state = False, dropout = 0.2, recurrent_dropout = 0.2)(x)
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

def model_train(train_data, train_mask, train_labels, write_dir, model, model_number, lr1, lr2, lr3):
    """ Trains n different models using different train and validation splittings"""
    for model_number in range(model_number):
        data, mask, labels, val_data, val_mask, val_labels = dataset_split(dataset_data=train_data, dataset_mask=train_mask, dataset_labels=train_labels, percent = 0.8, shuffling = True)
        history, sequence_model = run_experiment(initial_learning_rate = lr1, load_weights = False, early = True, model_number = model_number, model_basis = model, train_data=data, train_mask=mask, train_labels=labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, write_dir = write_dir)
        history, sequence_model = run_experiment(initial_learning_rate = lr2, load_weights = True, early = True, model_number = model_number, model_basis = model, train_data=data, train_mask=mask, train_labels=labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, write_dir = write_dir)
        history, sequence_model = run_experiment(initial_learning_rate = lr3, load_weights = True, early = True, model_number = model_number, model_basis = model, train_data=data, train_mask=mask, train_labels=labels, val_data=val_data, val_mask=val_mask, val_labels=val_labels, test_data=test_data, test_mask=test_mask, test_labels=test_labels, write_dir = write_dir)
    
    return history, sequence_model
    

def for_active_learning (confusion_matrix_data):
    false_positives = []
    false_negatives = []
    n,m = confusion_matrix_data.shape
    fn = 0
    for i in range(n):
        sum_fp = 0
        for j in range(m):
            if i == j:
                false_negatives.append(round(1-confusion_matrix_data[i,j],2))
            if i != j:
                sum_fp += confusion_matrix_data[j,i]
        false_positives.append(round(sum_fp,2))
    
    data = np.array([false_positives, false_negatives])
    summary = np.sum(data, axis=0)
    share_norm = [round(i/sum(summary),2) for i in summary]
    return share_norm

def active_learning_select(data, mask, labels, category_array, new_sampling_sizes):
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

def active_learning(train_data, train_mask, train_labels, val_data, val_mask, val_labels, test_data, test_mask, test_labels, category_array, write_dir, model, model_number_1 =3, model_number_2 =3, lr1_1 = 1e-3, lr2_1 = 1e-4, lr3_1 = 2e-5, lr1_2 = 1e-5, lr2_2 = 5e-6, lr3_2 = 2e-6, kappa = 0.2):
    global data, mask, labels, category_arr
    print (f': data_shape: {train_data.shape} data_mask_shape: {train_mask.shape} data_labels_shape: {train_labels.shape} val_data_shape: {val_data.shape} val_mask_shape: {val_mask.shape} val_labels_shape: {val_labels.shape} test_data_shape: {test_data.shape} test_mask_shape: {test_mask.shape} test_labels_shape: {test_labels.shape}')
    history, model = model_train_new (dataset_data = train_data, dataset_mask  = train_mask, dataset_labels = train_labels, write_dir=write_dir, model = model, model_number = model_number_1, lr1 = lr1_1, lr2 = lr2_1, lr3 = lr3_1, load_weights = False)
    with open (r'Active_learning.txt','a') as f:
        f.write('Iteration of active learning,  Accuracies, Dataset length\n')
        for iter in range(10):
            print (f': data_shape: {train_data.shape} data_mask_shape: {train_mask.shape} data_labels_shape: {train_labels.shape} val_data_shape: {val_data.shape} val_mask_shape: {val_mask.shape} val_labels_shape: {val_labels.shape} test_data_shape: {test_data.shape} test_mask_shape: {test_mask.shape} test_labels_shape: {test_labels.shape}')
            history, model = model_train_new (dataset_data = train_data, dataset_mask  = train_mask, dataset_labels = train_labels, write_dir=write_dir, model = model, model_number = model_number_2, lr1 = lr1_2, lr2 = lr2_2, lr3 = lr3_2, load_weights = True)
            cat_dct = predictions_voting(write_dir = write_dir, val=[test_data, test_mask], val_labels=test_labels, current_model=model, model_number = model_number_2)
            accuracies = accuracy_classes(cat_dct, category_type='category')
            print(f'Iteration of active learning: {iter}  Accuracies: {accuracies} Dataset length: {len(data)}')
            f.write(f'{iter},{accuracies},{len(data)}\n')
            false = [1-i for i in accuracies]
            false = false/sum(false)
            new_sampling_sizes = (len(part2_labels)*kappa*false).tolist()
            new_sampling_sizes = [int (i) for i in new_sampling_sizes]
            data, mask, labels, category_arr = active_learning_select(data=data, mask=mask, labels=labels, category_array=category_arr, new_sampling_sizes=new_sampling_sizes)


# Evaluation functions

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

def predictions(val, val_labels, sequence_model):
    ''' Prediction calulations '''
    Y_preds_category = sequence_model.predict(val).argmax(axis=-1)
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

def learning_graphics(metrics, xlim = range(0, 27), ylim = [0,1]):
    ''' Graphics for model training'''
    metrics_value = history.history[metrics]
    val_metrics_value = history.history['val_'+metrics]
    plt.figure()
    plt.plot(history.epoch, metrics_value, "r", label="Обучение")
    plt.plot(history.epoch, val_metrics_value, "b", label="Валидация")
    plt.title('Обучение и валидация: '+ metrics)
    plt.xlabel('Эпоха')
    plt.xticks(xlim)
    plt.ylabel('Значение '+ metrics)
    plt.ylim(ylim)
    plt.legend()
    plt.show()

def evaluation_plot(category_type='category'):
    ''' Prints confusion matrics for training'''
    target_category = cat_dct['target_categories_'+category_type]
    print(target_category)
    Y_preds = cat_dct['Y_preds_'+category_type]
    Y_labels = cat_dct['Y_labels_'+category_type]

    skplt.metrics.plot_confusion_matrix([target_category[i] for i in Y_labels], [target_category[i] for i in Y_preds],
                                    normalize=True,
                                    title="Матрица ошибок",
                                    cmap="Blues",
                                    hide_zeros=True,
                                    figsize=(5,5)
                                    );
    plt.xticks(rotation=90);


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


def confus_matr(normalize, cat_dct, category_type='category'):
    ''' Calculates confusion matrics and returns it in the form of array'''
    target_category = cat_dct['target_categories_'+category_type]
    print(target_category)
    Y_preds = cat_dct['Y_preds_'+category_type]
    Y_labels = cat_dct['Y_labels_'+category_type]

    confusion_matrix_data = confusion_matrix(y_true = [target_category[i] for i in Y_labels], y_pred = [target_category[i] for i in Y_preds], normalize=normalize)
    confusion_matrix_data = np.round(confusion_matrix_data,2)
    return confusion_matrix_data

# V. Model optimization

os.chdir(working_dir) # Returns to working directory
os.getcwd()
write_cat = 'model_1' # Directory that will contain the weights of optimized models 

df = pd.read_csv(working_dir+'/'+dir_united+'/'+dir_united+'.csv', sep = ',')
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["tag"]))
print(label_processor.get_vocabulary()) # Returs labels dictionary.

dataset_data, dataset_mask, dataset_labels = data_preparing(files_dir = files_dir, processing_suffix='_features') # Prepares features arrays
part1_data, part1_mask, part1_labels, part2_data, part2_mask, part2_labels = dataset_split(dataset_data=dataset_data, dataset_mask=dataset_mask, dataset_labels=dataset_labels, percent = 0.5, shuffling = True) # Splits dataset into two parts - for optimization and active learning
train_data, train_mask, train_labels, test_data, test_mask, test_labels = dataset_split(dataset_data=part1_data, dataset_mask=part1_mask, dataset_labels=part1_labels, percent = 0.8, shuffling = True) # Splits model to train and test parts; test parts will the same for all models. 

os.chdir(working_dir) # Returns to working directory
os.getcwd()

if not os.path.exists(write_cat):
    os.mkdir(write_cat)
write_dir = write_cat+'/'+ subdir
history, sequence_model = model_train(train_data=train_data, train_mask=train_mask, train_labels=train_labels, write_dir=write_dir, model=model_1, model_number = model_number, lr1=1e-3, lr2=1e-4, lr3=2e-5) # Optimizes n models for a given type of model and stores their weights in write_dir
cat_dct = predictions_voting(write_dir=write_dir, val=[test_data, test_mask], val_labels=test_labels, current_model = model_1, model_number = model_number) # Calculate model predictions for the ensemble of trained models (using modes of predictions).

accuracies = accuracy_classes(cat_dct=cat_dct, category_type='category') # Computes accuracies using the given ensemble of trained models

with open('Accuracies_different_models.csv','a') as f:
    f.write(f'{accuracies},{write_dir}\n')  # Writes file containing accuracies values for the given ensemble of trained models
