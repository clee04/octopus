from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pickle

var_dict = pickle.load(open('./savedir/conv_task/run_14_model_stats.pkl','rb'))['var_dict']
combined = OrderedDict()

for filt in range(3):
	min_val = np.min(var_dict['conv2_filter{}'.format(filt)][0])
	max_val = np.max(var_dict['conv2_filter{}'.format(filt)][0])

	fig, axes = plt.subplots(figsize=(8,7), nrows=4, ncols=4)
	for i,ax in enumerate(axes.flat):
	    im = ax.imshow(var_dict['conv2_filter{}'.format(filt)][0][:,:,i], vmin=min_val, vmax=max_val, cmap='inferno')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	plt.title('conv2_filter{}'.format(filt))
	plt.savefig('./evo_filter{}.png'.format(filt))

	combined['filt{}'.format(filt)] = [var_dict['conv2_filter{}'.format(filt)][0]]


var_dict = pickle.load(open('./savedir/conv_task/run_103_model_stats.pkl','rb'))['var_dict']
for filt in range(3):
	min_val = np.min(var_dict['conv2_filter{}'.format(filt)][0])
	max_val = np.max(var_dict['conv2_filter{}'.format(filt)][0])

	fig, axes = plt.subplots(figsize=(8,7), nrows=4, ncols=4)
	for i,ax in enumerate(axes.flat):
	    im = ax.imshow(var_dict['conv2_filter{}'.format(filt)][0][:,:,i], vmin=min_val, vmax=max_val, cmap='inferno')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	plt.title('conv2_filter{}'.format(filt))
	plt.savefig('./evo2_filter{}.png'.format(filt))

	combined['filt{}'.format(filt)].append(var_dict['conv2_filter{}'.format(filt)][1])


filters = pickle.load(open('./savedir/conv_task_tf/run_9_model_stats.pkl','rb'))['weight']
for filt in range(3):
	min_val = np.min(filters[:,:,:,filt])
	max_val = np.max(filters[:,:,:,filt])

	fig, axes = plt.subplots(figsize=(8,7), nrows=4, ncols=4)
	for i,ax in enumerate(axes.flat):
	    im = ax.imshow(filters[:,:,i,filt], vmin=min_val, vmax=max_val, cmap='inferno')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	plt.title('conv2_filter{}'.format(filt))
	plt.savefig('./conv_filter{}.png'.format(filt))

	combined['filt{}'.format(filt)].append(filters[:,:,:,int(filt)])


filters = pickle.load(open('./savedir/conv_task_tf/run_8_model_stats.pkl','rb'))['weight']
for filt in range(3):
	min_val = np.min(filters[:,:,:,filt])
	max_val = np.max(filters[:,:,:,filt])

	fig, axes = plt.subplots(figsize=(8,7), nrows=4, ncols=4)
	for i,ax in enumerate(axes.flat):
	    im = ax.imshow(filters[:,:,i,filt], vmin=min_val, vmax=max_val, cmap='inferno')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	plt.title('conv2_filter{}'.format(filt))
	plt.savefig('./conv2_filter{}.png'.format(filt))

	combined['filt{}'.format(filt)].append(filters[:,:,:,filt])

for key, val in combined.items():
	for i in range(len(val)):
		for j in range(i+1,len(val)):
			print(i,j,'\t',np.mean(np.square(val[i]-val[j])))
	print()
