"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import io
import json
import pickle
import time
import numpy as np
np.set_printoptions(precision=3)
from stimulus import Stimulus
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from parameters import *
from model_util import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

"""
This model is trained on the colorization task using backpropagation on
the same inputs as the model trained using genetic evolutionary algorithm.
* Input: latent representation from the second to last layer of a convolutional 
         autoencoder that was trained on reproducing a grayscale image.
* Label: groundtruth colored images
"""
class ConvModelTop():
    def __init__(self, input_data, target_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()
 
    def run_model(self):
        # Run model
        # conv = tf.layers.conv2d(inputs=self.input_data, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)     # (64,64,128)
        # upsample = tf.image.resize_images(conv, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)             # (128,128,128)
        output = tf.layers.conv2d(inputs=self.input_data, filters=3, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='filters')
        self.output = tf.multiply(tf.nn.relu(tf.reshape(output, [par['batch_train_size'],par['n_output']])),1,name='output')     # (128,128,16)
 
    def optimize(self):
        # Calculate loss and optimize
        self.loss = tf.multiply(tf.losses.mean_squared_error(self.target_data, self.output), 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)

def main(gpu_id = None):

    # Select gpu
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],*par['n_input']],name='input')
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']],name='target')

    # Generate stimulus
    stim = Stimulus()

    # Model stats
    losses = []
    testing_losses = []
    save_iter = []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = ConvModelTop(x,y)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        prev_loss = 1000000
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
                    test_loss, test_output = sess.run([model.loss, model.output], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    # Plot model outputs
                    if test_loss < prev_loss:
                        prev_loss = test_loss
                        plot_conv_outputs(target_data, model_output, test_target, test_output, i)

                        # Save training stats and model
                        weight = sess.run(tf.get_collection(tf.GraphKeys.VARIABLES, 'filters/kernel')[0])
                        pickle.dump({'iter':save_iter,'weight':weight,'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                        
                        saved_path = saver.save(sess, './conv_model_top')
                        print('model saved in {}'.format(saved_path))

                    # Stop training
                    if test_loss < 50:
                        break

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
        # Set run parameters
        updates = {
            'note'              : 'training conv top on while saving filter',
            'task'              : 'conv_task_tf',
            'run_number'        : 9
        }

        # Save updated parameters
        update_parameters(updates)
        with io.open(par['save_dir']+'run_'+str(par['run_number'])+'_params.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(updates,
                              indent=4, sort_keys=False,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))

    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))





















