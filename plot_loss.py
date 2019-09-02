import numpy as np
import matplotlib.pyplot as plt
import pickle

ff = pickle.load(open('./savedir/bw1_to_color/run_999_model_stats.pkl','rb'))
ff_iter = ff['iter']
ff_loss = ff['losses']
min_len = len(ff_loss)

conv = pickle.load(open('./savedir/bw1_to_color/run_1002_model_stats.pkl','rb'))
conv_iter = conv['iter']
conv_loss = conv['losses']
min_len = min(min_len, len(conv_loss))
min_len = 500

plt.plot(ff_iter[:min_len], ff_loss[:min_len], label='Fully Connected Feedforward Network')
plt.plot(conv_iter[:min_len], conv_loss[:min_len], label='Convolutional Autoencoder')
plt.legend(loc='upper right')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss (mean squared error)')
plt.show()


# gen = pickle.load(open('./savedir/conv_task/run_23_model_stats.pkl','rb'))
# # gen_iter = gen['iter']
# gen_loss = gen['losses']
# gen_iter = [i for i in range(len(gen_loss))]
# min_len = len(gen_loss)

# conv = pickle.load(open('./savedir/conv_task_tf/run_9_model_stats.pkl','rb'))
# conv_iter = conv['iter']
# conv_loss = conv['losses']
# min_len = min(min_len, len(conv_loss))
# min_len = 155

# plt.plot(gen_iter, gen_loss, label='Trained using genetic algorithm')
# plt.plot(conv_iter[:min_len], conv_loss[:min_len], label='Trained using backpropagation')
# plt.legend(loc='upper right')
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Loss (mean squared error)')
# plt.show()

