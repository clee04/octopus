import tensorflow as tf
import pickle
import time
import numpy as np
from stimulus import Stimulus
from parameters import *
from model_util import *
from evo_utils import *
from evo_model import EvoModel

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
This function loads trained convolutional model and 
evolutionary models and saves their output images
"""
def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()

    # Load saved genetically trained model
    evo_model = EvoModel()
    saved_evo_model = pickle.load(open('./savedir/conv_task/run_21_model_stats.pkl','rb'))
    evo_model.update_variables(saved_evo_model['var_dict'])
    print('Loaded evo model')

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # Load trained convolutional autoencoder
            if par['task'] == 'conv_task':
                folder = './latent_all_img_batch16_filt16_loss80/'
                conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
                conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
                print('Loaded conv model from',folder)

                # Generate batch and save output images
                for i in range(4):
                    input_data, conv_target, evo_target = stim.generate_train_batch()

                    # Run input through convolutional model
                    feed_dict = {'x:0': input_data, 'y:0': conv_target}
                    test_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

                    # Run latent output through evolutionary model
                    evo_model.load_batch(encoded, evo_target)
                    evo_model.run_models()
                    evo_model.judge_models()

                    # Save output from both models
                    plot_conv_evo_outputs(conv_target, conv_output, evo_target, evo_model.output, i, test=True)
            else:
                folder = './'
                conv_top = tf.train.import_meta_graph(folder + 'conv_model_top.meta', clear_devices=True)
                conv_top.restore(sess, tf.train.latest_checkpoint(folder)) 
                
                for i in range(4):
                    test_input, test_target, _ = stim.generate_test_batch()
                    feed_dict = {'input:0': test_input, 'target:0': test_target}
                    test_loss, test_output = sess.run(['l:0', 'output:0'], feed_dict=feed_dict)
                    plot_conv_all(test_target, test_output, i)


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
            'a_note'            : 'latent to evo, all img',
            'run_number'        : 0,
            'task'              : 'conv_task_tf'
        }
        update_parameters(updates)
        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))







