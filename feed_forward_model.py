"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import os, io, json
import cv2
import pickle
import time
import numpy as numpy
from stimulus import Stimulus
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from parameters import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
        
        # Get weights and biases
        if par['num_layers'] == 5:
            with tf.variable_scope('encoder'):
                self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
                self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)
                self.W_enc = tf.get_variable('W_enc', initializer=par['W_enc_init'], trainable=True)
                self.b_latent = tf.get_variable('b_latent', initializer=par['b_latent_init'], trainable=True)
                self.W_link = tf.get_variable('W_link', initializer=par['W_link_init'], trainable=True)
                self.b_link = tf.get_variable('b_link', initializer=par['b_link_init'], trainable=True)

            with tf.variable_scope('decoder'):
                self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
                self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
                self.W_link2 = tf.get_variable('W_link2', initializer=par['W_link2_init'], trainable=True)
                self.b_link2 = tf.get_variable('b_link2', initializer=par['b_link2_init'], trainable=True)
                self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
                self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)
        
        if par['num_layers'] == 3:
            with tf.variable_scope('encoder'):
                self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
                self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)
                self.W_enc = tf.get_variable('W_enc', initializer=par['W_enc_init'], trainable=True)
                self.b_latent = tf.get_variable('b_latent', initializer=par['b_latent_init'], trainable=True)

            with tf.variable_scope('decoder'):
                self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
                self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
                self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
                self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

        elif par['num_layers'] == 2:
            with tf.variable_scope('encoder'):
                self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
                self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)

            with tf.variable_scope('decoder'):
                self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
                self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
                self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
                self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

        # Run input through the model layers
        if par['num_layers'] == 5:
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.link = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
            self.latent = tf.nn.sigmoid(tf.add(tf.matmul(self.link, self.W_link), self.b_link))
            self.link2 = tf.nn.sigmoid(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
            self.dec = tf.nn.relu(tf.add(tf.matmul(self.link2, self.W_link2), self.b_link2))
        
        elif par['num_layers'] == 3:
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.latent = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
        
        self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))

    def optimize(self):
        # Calculae loss
        self.loss = tf.reduce_mean(tf.square(self.target_data - self.output))
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']])
    
    # Model stats
    losses = []
    testing_losses = []
    save_iter = []
    prev_loss = 10000

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x,y)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, target_data, _ = stim.generate_train_batch()
            feed_dict = {x: input_data, y: target_data}
            _, train_loss, model_output = sess.run([model.train_op, model.loss, model.output], feed_dict=feed_dict)


            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, train_loss, time.time()-start))
                losses.append(train_loss)
                save_iter.append(i)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:

                    # Generate batch from testing set and check the output
                    test_input, test_target, _ = stim.generate_test_batch()
                    feed_dict = {x: test_input, y: test_target}
                    test_loss, test_output = sess.run([model.loss, model.output], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    if test_loss < prev_loss:
                        prev_loss = test_loss
                    
                        # Results from a training sample
                        original1 = target_data[0].reshape(par['out_img_shape'])
                        output1 = model_output[0].reshape(par['out_img_shape'])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(original1,'Training',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

                        # Results from a testing sample
                        original2 = test_target[1].reshape(par['out_img_shape'])
                        output2 = test_output[1].reshape(par['out_img_shape'])
                        original3 = test_target[2].reshape(par['out_img_shape'])
                        output3 = test_output[2].reshape(par['out_img_shape'])
                        cv2.putText(original2,'Testing',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                    
                        vis1 = np.concatenate((original1, output1), axis=1)
                        vis2 = np.concatenate((original2, output2), axis=1)
                        vis3 = np.concatenate((original3, output3), axis=1)
                        vis = np.concatenate((vis1, vis2), axis=0)
                        vis = np.concatenate((vis, vis3), axis=0)

                        cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)

                        pickle.dump({'iter':save_iter,'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))


                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()

                # Stop training if loss is small enough (just for sweep purposes)
                if train_loss < 30:
                    break


if __name__ == "__main__":
    try:
        gpu_id = sys.argv[1]
    except:
        gpu_id = None

    # Run Model
    t0 = time.time()
    try:
        # Set run parameters
        updates = {
            'note'              : 'ff 3 layer on colorization',
            'task'              : 'bw1_to_color',
            'run_number'        : 999
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














