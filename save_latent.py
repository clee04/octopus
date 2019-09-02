"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import numpy as np
import os
import time
from stimulus import Stimulus
from parameters import *
from evo_utils import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


"""
This function loads a trained convolutional autoencoder
and saves the latent representation output from the model
to be used for training other models on the colorization task
"""
def main(gpu_id = None):

    # Reset Tensorflow graph
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        # Load trained convolutional autoencoder
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            folder = './upsample2/'
            conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
            conv_model.restore(sess, tf.train.latest_checkpoint(folder))
            print('Loaded model from',folder)

        # Get all images from dataset
        stim = Stimulus()
        train_filename, test_filename, train_data, test_data, dummy_output = stim.get_all_data()

        # Run through all images in the dataset and save latent output
        save_latent(train_filename, train_data, dummy_output, sess)
        save_latent(test_filename, test_data, dummy_output, sess)



def save_latent(filename, data, dummy_output, sess):
    # Run images through the convolutional autoencoder and 
    # save the latent representation to be used for later training
    while len(data) > 0:
        # Get a batch of data
        b = par['batch_train_size']
        input_data = data[:b]

        # Run the batch through the model
        feed_dict = {'x:0': input_data, 'y:0': dummy_output}
        conv_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)
        
        # Save the latent representation from the model
        for n in range(len(encoded)):
            name = filename[n]
            output = encoded[n]
            np.save('./inner_latent2/'+name, output)

        if len(data) == b:
            break
        elif len(data) < b*2:
            filename = filename[-b:]
            data = data[-b:]
        else:
            filename = filename[b:]
            data = data[b:]


if __name__ == "__main__":
    # GPU
    try:
        gpu_id = sys.argv[1]
    except:
        gpu_id = None

    # Run Model
    t0 = time.time()
    try:
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))





















