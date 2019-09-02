"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import pickle
import time
import numpy as np
from stimulus import Stimulus
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from parameters import *
from model_util import *

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

"""
This is a convolutional autoencoder trained to reproduce grayscale
images. The latent representation from the second to last layer is 
later used to train other models on the colorization task.
* Input: grayscale images
* Label: grayscale images
"""
class Model:

    def __init__(self, input_data, target_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()

 
    def run_model(self):

        x = tf.reshape(self.input_data, shape=[par['batch_train_size'],*par['inp_img_shape'],1])

        # Encoding
        conv1    = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
        conv2    = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        conv3    = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')

        # Decoding
        bottleneck = tf.image.resize_images(maxpool3, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(inputs=bottleneck, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)               # (64,64,64)
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)     # (64,64,128)
        upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)             # (128,128,128)
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)     # (128,128,16)
        self.latent = tf.multiply(conv6, 1, name='encoded')

        conv7 = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='filters')
        self.output = tf.multiply(tf.nn.relu(tf.reshape(conv7, [par['batch_train_size'],par['n_output']])), 1, name='o')
 
    def optimize(self):
        # Calculae loss and optimize
        self.loss = tf.multiply(tf.losses.mean_squared_error(self.target_data, self.output), 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']], name='x')
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']], name='y')

    # Generate stimulus
    stim = Stimulus()

    # Model stats
    losses = []
    testing_losses = []
    save_iter = []

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x,y)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Train the model
        prev_loss = 10000000
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training batch and train model
            input_data, target_data, _ = stim.generate_train_batch()
            feed_dict = {x: input_data, y: target_data}
            _, train_loss, model_output = sess.run([model.train_op, model.loss, model.output], feed_dict=feed_dict)

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print_conv_stats(i, train_loss, time.time()-start)
                losses.append(train_loss)
                save_iter.append(i)

                # Test and save model
                if i % par['save_iter'] == 0:

                    # Generate test bach and get model performance
                    test_input, test_target, _ = stim.generate_test_batch()
                    feed_dict = {x: test_input, y: test_target}
                    test_loss, test_output = sess.run([model.loss, model.output,], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    # Plot model outputs
                    if test_loss < prev_loss:
                        prev_loss = test_loss
                        plot_conv_outputs(target_data, model_output, test_target, test_output, i)

                        # Save training stats and model
                        weight = sess.run(tf.get_collection(tf.GraphKeys.VARIABLES, 'filters/kernel')[0])
                        pickle.dump({'weight':weight,'iter':save_iter,'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                        
                        saved_path = saver.save(sess, './upsample2/conv_model_with_latent')
                        print('model saved in {}'.format(saved_path))

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()


if __name__ == "__main__":
    # GPU
    try:
        gpu_id = sys.argv[1]
    except:
        gpu_id = None

    # Run Model
    t0 = time.time()
    try:
        updates = {
            'a_note'            : 'simpler conv model on colorization saving weights',
            'task'              : 'bw1_to_color',
            'run_number'        : 1003
        }
        update_parameters(updates)
        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))










