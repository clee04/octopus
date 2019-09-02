Octopus Camoflage Simulation
============================
This project is an effort to simulate octopus's ability to camoflage in color despite only being able to see in black and white.
A live demo using computer camera and screen can be ran by typing command "python3 video.py" from the code folder, although the 
colorization performance of the model is compromised given that the model was trained only on underwater images.

Preprocessing Data
==================
*Dataset may be shared upon request and with consent from Professor Rick Stevens of University of Chicago.
- app.py: converts colored images to grayscale images; used to build black and white input images from colored image dataset
- crop.sh: script that was used to crop large images from original dataset to create smaller sized images that were used to train models
- stimulus.py: loads the dataset and generate train and testing batches for the appropriate task type

Model Architectures
===================
- train_conv_model.py: convolutional autoencoder architecture; can be trained to produce black & white images or colored images
- feed_forward_model.py: model architecture for the fully connected three/five layer feed forward network
- evo_model.py: contains a model class that is trained using genetic algorithm
- best_evo.py: a light-weight version of the evolutionary network model that runs assuming 1 network and 1 input image, used by video.py
- train_conv_top.py: takes saved latent representaion as an input and trains the last layer of the convolutional autoencoder using backpropagation

Utility Files
=============
- parameters.py: list of parameters that can be changed when training and testing the model
- model_util.py: helper functions for various network models
- evo_utils.py: helper functions for the evolutionary model
- video.py: code that takes input from the camera, runs through saved models, and displays the output on screen

Analysis Files
==============
- load_and_graph.py: loads up a trained model and plots output images on several randomly picked images
- save_latent.py: loads a trained convolutional model and saves its latent representation for each image to be used for other models
- plot_loss.py: plots a loss curve from saved model training history
- weight_analysis.py: plots the filters for the models that were trained for colorization and compare the distance between those models
