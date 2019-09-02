"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""
import tensorflow as tf
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from collections import OrderedDict
import io, time, itertools
import json, pickle, copy
import numpy as np
np.set_printoptions(precision=3)

from stimulus import Stimulus
from parameters import *
from model_util import *
from evo_utils import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


"""
This model is trained on the colorization task using genetic evolutionary algorithm.
* Input: latent representation from the second to last layer of a convolutional 
         autoencoder that was trained on reproducing a grayscale image.
* Label: groundtruth colored images
"""
class EvoModel:

    def __init__(self):
        
        self.make_constants()
        self.make_variables()
        self.original = True
 
    # Make variables that will be trained (weights and biases)
    def make_variables(self):
        self.var_dict = OrderedDict()
        for i in range(3):
            self.var_dict['conv2_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,par['num_conv1_filters'])).astype(cp.float32)

        # self.var_dict['conv2_bias'] = cp.random.normal(size=(par['n_networks'],*par['inp_img_shape'],3)).astype(cp.float32)
        self.var_dict['b_out'] = cp.random.normal(size=(par['n_networks'],*par['out_img_shape'])).astype(cp.float32)

    # Constants that are not trained
    def make_constants(self):
        constants = ['n_networks','mutation_rate','mutation_strength','cross_rate']

        self.con_dict = {}
        for c in constants:
            self.con_dict[c] = to_gpu(par[c])

    # Used for getting variables from a saved model
    def update_variables(self, updates):
        for key, val in updates.items():
            self.var_dict[key] = to_gpu(val)

    # Used for getting constants from a saved model
    def update_constant(self, name, val):
        self.con_dict[name] = to_gpu(val)

    # Get input and target data
    def load_batch(self, input_data, target_data):
        self.input_data = cp.repeat(cp.expand_dims(to_gpu(input_data), axis=0), par['n_networks'], axis=0)
        self.target_data = to_gpu(target_data)

    # Run all networks
    def run_models(self):
        conv = relu(convolve(self.input_data, self.var_dict)) + cp.expand_dims(self.var_dict['b_out'], axis=1)) #+ cp.expand_dims(self.var_dict['conv2_bias'], axis=1))
        self.output = cp.reshape(conv, (par['n_networks'],par['batch_train_size'],*par['out_img_shape']))

    # Calculate loss for all networks, sort the networks based on their loss values
    def judge_models(self):
        img_len = par['img_size']
        self.target_data = cp.reshape(self.target_data, (par['batch_train_size'],*par['out_img_shape']))
        target = cp.repeat(cp.expand_dims(self.target_data,axis=0),par['n_networks'],axis=0)
        
        # self.loss contains mean squared error for each network
        self.loss = cp.mean(cp.square(target - self.output),axis=(1,2,3,4)).astype(cp.float64)
        self.output = cp.reshape(self.output, (par['n_networks'],par['batch_train_size'],par['n_output']))
        self.rank = cp.argsort(self.loss).astype(cp.int16)

        if self.original:
            self.original = False
        else:
            # If not first generation, pick one network to survive for each parental line
            replace_parents = cp.zeros(par['num_survivors']).astype(cp.int16)
            for i in range(par['num_survivors']):
                replace_parents[i] = cp.argmin(self.loss[cp.arange(i,par['n_networks'],par['num_survivors'])])*par['num_survivors'] + i
            
            temp = copy.deepcopy(replace_parents)
            for i,idx in enumerate(cp.argsort(self.loss[replace_parents])):
                replace_parents[i] = temp[idx]

            self.rank = (cp.ones(par['n_networks'])*(par['n_networks']-1)).astype(cp.int16)
            self.rank[:par['num_survivors']] = replace_parents
            
            # Allow at least one among newly introduced networks (migrators) to survive
            best_migrator = par['n_networks'] - par['num_migrators'] + cp.argmin(self.loss[-par['num_migrators']:].astype(cp.float64)).astype(cp.int16)
            if par['num_migrators'] > 0 and best_migrator not in self.rank[:par['num_survivors']]:
                self.rank[par['num_survivors']-1] = best_migrator

        # Sort based on the rank
        for name in self.var_dict.keys():
            self.var_dict[name][:par['num_survivors']] = self.var_dict[name][self.rank[:par['num_survivors']],...]

    # Get losses, default is sorted by performance
    def get_losses(self, ranked=True):
        if ranked:
            return to_cpu(self.loss[self.rank])
        else:
            return to_cpu(self.loss)

    # Decrease mutation rate and strength; reset flag returns the values to default values
    def slowdown_mutation(self, reset=False):
        if reset:
            self.con_dict['mutation_rate'] = min(0.6, self.con_dict['mutation_rate'])
            self.con_dict['mutation_strength'] = min(0.45, self.con_dict['mutation_strength'])
        else:
            self.con_dict['mutation_rate'] = max(0.1, self.con_dict['mutation_rate'] * 0.7)
            self.con_dict['mutation_strength'] = max(0.05, self.con_dict['mutation_strength'] * 0.8)

    # Increase mutation rate and strength
    def speed_up_mutation(self):
        self.con_dict['mutation_rate'] = min(0.7, self.con_dict['mutation_rate'] * 1.25)
        self.con_dict['mutation_strength'] = min(0.45, self.con_dict['mutation_strength'] * 1.125)

    # Generate new models by introducing random mutation to best/selected models
    def breed_models_genetic(self):
        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0],\
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

    # Replace last few models with newly newly generated random models
    def migration(self):
        for key in self.var_dict.keys():
            shape = self.var_dict[key][-par['num_migrators']:,...].shape
            self.var_dict[key][-par['num_migrators']:,...] = cp.random.normal(size=shape).astype(cp.float32)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Generate stimulus
    stim = Stimulus()

    # Initialize evolutionary model
    evo_model = EvoModel()

    # Model stats
    losses = []
    testing_losses = []
    save_iter = []

    # Reset Tensorflow graph
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # Load saved convolutional autoencoder
            folder = './latent_all_img_batch16_filt16_loss80/'
            conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
            conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
            print('Loaded model from',folder)


        stuck = 0
        test_loss = [1000000]
        threshold = [10000, 5000, 1000, 750, 500, 300, 150, -1]

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate train batch and run through the convolutional autoencoder
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {'x:0': input_data, 'y:0': conv_target}
            conv_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

            # One cycle of evolutionary model
            evo_model.load_batch(encoded, evo_target)
            evo_model.run_models()
            evo_model.judge_models()
            evo_model.breed_models_genetic()

            # If all models are performing poorly, introduce new randomly generated models
            evo_loss = evo_model.get_losses(True)
            if par['num_migrators']:
                evo_model.migration()
            if evo_loss[0] < 10000:
                par['num_migrators'] = 0

            # Decrease mutation rate when loss value is below certain threshold
            if len(threshold) > 0 and evo_loss[0] < threshold[0]:
                evo_model.slowdown_mutation()
                threshold.pop(0)

            # If there is no improvement in performance for many iterations, change mutation rate
            if evo_loss[0] < test_loss[0]:
                stuck = 0
            else:
                stuck += 1
                if stuck > 20:
                    # evo_model.speed_up_mutation()
                    evo_model.slowdown_mutation()
                    stuck = 0

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print_evo_stats(i, evo_model.con_dict['mutation_rate'], evo_model.con_dict['mutation_strength'], stuck, conv_loss, np.array([*evo_loss[0:3],evo_loss[par['num_survivors']-1]]), time.time()-start)
                losses.append(evo_loss[0])
                save_iter.append(i)

                # Save model and output
                if i % par['save_iter'] == 0 and (evo_loss[0] < test_loss[0] or i%100 == 0):

                    # Generate batch from testing set and run through both models
                    input_data, conv_target, evo_target = stim.generate_test_batch()
                    feed_dict = {'x:0': input_data, 'y:0': conv_target}
                    test_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

                    evo_model.load_batch(encoded, evo_target)
                    evo_model.run_models()
                    evo_model.judge_models()

                    test_loss = evo_model.get_losses(True)
                    testing_losses.append(test_loss[0])

                    # Save output images
                    plot_conv_evo_outputs(conv_target, conv_output, evo_target, evo_model.output, i)

                    # Save model
                    pickle.dump({'iter':save_iter,'var_dict':evo_model.var_dict, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                        open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
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
            'a_note'            : 'latent to evo, raw_im1 dataset, lower mutation rate',
            'task'              : 'conv_task',
            'run_number'        : 103
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





















