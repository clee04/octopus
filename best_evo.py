"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""
from parameters import *
from collections import OrderedDict
import numpy as np

def relu(x):
    return np.maximum(0., x, dtype=x.dtype)

def pad(x):
    shape = x.shape
    img_size = shape[-2]

    temp = np.zeros((img_size+2,img_size+2,shape[-1]))
    temp[:img_size,:img_size,:] = x
    return temp

def apply_filter(x, filt):
    img_size = x.shape[-2] - 2

    temp = np.zeros((img_size,img_size))
    for i in range(img_size):
        for j in range(img_size):
            temp[i,j] = np.sum(x[i:i+3,j:j+3,:] * filt)
    
    return temp

def convolve(x, var_dict, filt_type):
    conv = np.zeros((*par['inp_img_shape'],3))
        
    i = 0
    x = pad(x)
    for key in var_dict.keys():
        if filt_type in key:
            conv[:,:,i] = apply_filter(x, var_dict[key])
            i += 1

    return conv

"""
Evolutionary model adapted for real-time demo
Assumes 1 network and batch size of 1
"""
class BestEvoModel:

    def __init__(self):
        
        self.var_dict = OrderedDict()
 
    def update_variables(self, updates):
        for key, val in updates.items():
            self.var_dict[key] = val

    def load_batch(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def run_models(self):
        conv1 = relu(convolve(self.input_data, self.var_dict, 'conv2_filter')) #+ self.var_dict['conv2_bias'])
        self.output = np.reshape(conv1, (par['out_img_shape']))



