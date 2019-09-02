import numpy as np
import os

print("Loading parameters...")

"""
Set independent parameters
"""
par = {
    # Setup parameters
    'save_dir'          : './savedir/',
    'input_dir'         : './bw_im2/',
    'target_dir'        : './raw_im2/',
    'img_size'          : 128,

    # Training parameters
    'task'              : 'conv_task',
    'simulation'        : False,
    'learning_rate'     : 0.0005, #default = 0.001
    'batch_train_size'  : 16,
    'num_iterations'    : 750001,
    'normalize01'       : False,
    'print_iter'        : 1,
    'save_iter'         : 5,
    'one_img'           : False,
    'run_number'        : 0,

    # Feedforward network shape
    'n_enc'             : 125,
    'n_link'            : 100,
    'n_latent'          : 75,
    'n_dec'             : 125,
    'num_layers'        : 5,
    
    # Convolutional network shape
    'num_conv1_filters' : 16,

    # Evolutionary network shape
    'n_networks'        : 65,
    'survival_rate'     : 0.08,
    'mutation_rate'     : 0.6,
    'mutation_strength' : 0.45,
    'migration_rate'    : 0.0,
    'cross_rate'        : 0.0,
    'use_crossing'      : False
}

"""
Set dependent parameters
"""
def update_dependencies():
    print('Updating dependencies...\n')

    if par['task'] == 'bw_to_bw':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])    # used to be (par['img_size'],par['img_size'],3)
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size'] #par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']*3 # par['n_input']
        par['save_dir'] = './savedir/bw_to_bw/'

    elif par['task'] == 'bw_to_bw_simple':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'])
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']
        par['save_dir'] = './savedir/bw_to_bw_simple/'

    elif par['task'] == 'bw3_to_color':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'],3)
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']
        par['save_dir'] = './savedir/bw3_to_color/'

    elif par['task'] == 'bw1_to_color':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']*3
        par['save_dir'] = './savedir/bw1_to_color/'

    elif par['task'] == 'conv_task':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']*3
        par['save_dir'] = './savedir/conv_task/'

    elif par['task'] == 'conv_task_tf':
        par['input_dir'] = './latent_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = (par['img_size'],par['img_size'],16)
        par['n_output'] = (par['img_size']*par['img_size']*3)
        par['save_dir'] = './savedir/conv_task_tf/'

    if par['simulation']:
        par['save_dir'] = './simulation/'

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate'])
    par['num_migrators'] = int(par['n_networks'] * par['migration_rate'])

    # Set up initializers
    # if par['num_layers'] == 5:
    #     par['W_in_init'] = np.float32(np.random.normal(size=[par['n_input'], par['n_enc']]))
    #     par['b_enc_init'] = np.float32(np.random.normal(size=(1,par['n_enc'])))

    #     par['W_enc_init'] = np.float32(np.random.normal(size=[par['n_enc'], par['n_link']]))
    #     par['b_latent_init'] = np.float32(np.random.normal(size=(1,par['n_link'])))

    #     par['W_link_init'] = np.float32(np.random.normal(size=[par['n_link'], par['n_latent']]))
    #     par['b_link_init'] = np.float32(np.random.normal(size=(1,par['n_latent'])))
        
    #     par['W_dec_init'] = np.float32(np.random.normal(size = [par['n_latent'], par['n_link']]))
    #     par['b_dec_init'] = np.float32(np.random.normal(size=(1,par['n_link'])))

    #     par['W_link2_init'] = np.float32(np.random.normal(size=[par['n_link'], par['n_dec']]))
    #     par['b_link2_init'] = np.float32(np.random.normal(size=(1,par['n_dec'])))

    #     par['W_out_init'] = np.float32(np.random.normal(size=[par['n_dec'], par['n_output']]))
    #     par['b_out_init'] = np.float32(np.random.normal(size=(1,par['n_output'])))
    
    # elif par['num_layers'] == 3:
    #     par['W_in_init'] = np.float32(np.random.normal(size=[par['n_input'], par['n_enc']]))
    #     par['b_enc_init'] = np.float32(np.random.normal(size=(1,par['n_enc'])))

    #     par['W_enc_init'] = np.float32(np.random.normal(size=[par['n_enc'], par['n_latent']]))
    #     par['b_latent_init'] = np.float32(np.random.normal(size=(1,par['n_latent'])))
        
    #     par['W_dec_init'] = np.float32(np.random.normal(size=[par['n_latent'], par['n_dec']]))
    #     par['b_dec_init'] = np.float32(np.random.normal(size=(1,par['n_dec'])))

    #     par['W_out_init'] = np.float32(np.random.normal(size=[par['n_dec'], par['n_output']]))
    #     par['b_out_init'] = np.float32(np.random.normal(size=(1,par['n_output'])))

"""
Update parameters based on the given dictionary (updates)
"""
def update_parameters(updates):
    print('Updating parameters...')

    for (key, val) in updates.items():
        par[key] = val

    # Update any dependent parameters
    update_dependencies()


# Run update dependencies when loading parameters
update_dependencies()
