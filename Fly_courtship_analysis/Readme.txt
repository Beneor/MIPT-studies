The following scripts can be used to train a video classifier, which predicts nine classes of Drosophila male behavior while courting a female.
Requires a dataset of labeled videos (not provided).
Training scripts:
1. data_prepare.py - prepares command file to perform extraction of spatial features using pretrainad convolutional neural network.
Requires a dataset of labeled videos (not provided). 
2. features_extraction_training.py - performes spatial features extraction.
3. optimization.py - performs training of models, including different types of recurrent neural network. 
Requires numpy arrays of extracted features obtained at the previous stage.
4. active_learning.py - performs training of models, using active learning.
5. active_learning_control.py - performs training of models without active learning (control).

The training algorithm is based on the following script:
Keras: Video Classification with a CNN-RNN Architecture (https://keras.io/examples/vision/video_classification/) 

The following software should be pre-installed:
jupyterlab 4.0.6
keras 2.13.1
matolotlib 3.8.0
moviepy
numpy 1.24.3
opencv-python 4.8.1.78
pandas 2.1.1
scipy 1.11.2
scikit-learn 1.3.1
tensorflow 2
