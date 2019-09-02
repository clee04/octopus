from parameters import *
import numpy as np
np.set_printoptions(precision=3)
import cv2

# Print current convolutional model statistics
def print_conv_stats(i, train_loss, time):
    print('Model {:2} | Task: {:s} | Iter: {:6} | Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
          par['run_number'], par['task'], i, train_loss, time))

# Print current evolutionary model statistics
def print_evo_stats(i, mutation_rate, mutation_strength, stuck, conv_loss, evo_loss, time):
    print('Model {:1} | Iter: {:4} | Mut Rate: {:.2f} | Mut Strength: {:.2f} | Stuck: {:2} | Conv Loss: {:.2f} | Evo Loss: {} | Run Time: {:5.3f}s'.format( \
          par['run_number'], i, mutation_rate, mutation_strength, stuck, conv_loss, evo_loss, time))

# Save convolutional model output images
def plot_conv_outputs(target_data, model_output, test_target, test_output, i):

    # One result from a training batch
    label1  = target_data[0].reshape(par['out_img_shape'])
    output1 = model_output[0].reshape(par['out_img_shape'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(label1,'Train',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    # Results from testing batch
    label2  = test_target[1].reshape(par['out_img_shape'])
    output2 = test_output[1].reshape(par['out_img_shape'])
    label3  = test_target[2].reshape(par['out_img_shape'])
    output3 = test_output[2].reshape(par['out_img_shape'])
    cv2.putText(label2,'Tesst',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    vis1 = np.concatenate((label1, output1), axis=1)
    vis2 = np.concatenate((label2, output2), axis=1)
    vis3 = np.concatenate((label3, output3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    vis = np.concatenate((vis, vis3), axis=0)

    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)

# Save convolutional model output images
def plot_conv_all(test_target, test_output, i):

    # Results from testing batch
    label1  = test_target[0].reshape(par['out_img_shape'])
    output1 = test_output[0].reshape(par['out_img_shape'])
    label2  = test_target[1].reshape(par['out_img_shape'])
    output2 = test_output[1].reshape(par['out_img_shape'])
    label3  = test_target[2].reshape(par['out_img_shape'])
    output3 = test_output[2].reshape(par['out_img_shape'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(label1,'Test',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    vis1 = np.concatenate((label1, output1), axis=1)
    vis2 = np.concatenate((label2, output2), axis=1)
    vis3 = np.concatenate((label3, output3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    vis = np.concatenate((vis, vis3), axis=0)

    cv2.imwrite('{}testing_run{}_{}.png'.format(par['save_dir'],par['run_number'],i),vis)

# Save convolutional and evolutionary model outputs
def plot_conv_evo_outputs(conv_target, conv_output, evo_target, evo_output, i, test=False):

    # Results from a training sample
    if test:
        num_plot = 1
    else:
        num_plot = 4

    for j in range(num_plot):
        outputs = []
        for b in range(4):
            batch = j*4 + b

            # One result from convolutional model
            label1  = conv_target[batch].reshape(par['out_img_shape'])
            output1 = conv_output[batch].reshape(par['out_img_shape'])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(label1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

            # Results from best three evolutionary model
            label2  = evo_target[batch].reshape(par['out_img_shape'])
            output2 = evo_output[0][batch].reshape(par['out_img_shape'])
            output3 = evo_output[1][batch].reshape(par['out_img_shape'])
            output4 = evo_output[2][batch].reshape(par['out_img_shape'])
            cv2.putText(label2,'Evo',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output3,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output4,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

            vis1 = np.concatenate((label1, output1), axis=1)
            vis2 = np.concatenate((label2, output2), axis=1)
            vis3 = np.concatenate((label2, output3), axis=1)
            vis4 = np.concatenate((label2, output4), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
            vis = np.concatenate((vis, vis3), axis=0)
            vis = np.concatenate((vis, vis4), axis=0)
            outputs.append(vis)

        if len(outputs) == 1:
            vis = outputs[0]
        else:
            vis = np.concatenate((outputs[0],outputs[1]), axis=1)
            for batch in range(2,len(outputs)):
                vis = np.concatenate((vis,outputs[batch]), axis=1)
        
        if test:
            cv2.imwrite('{}testing_run{}_{}_{}.png'.format(par['save_dir'],par['run_number'],j,i), vis)
        else:
            cv2.imwrite('{}run_{}_test_{}.png'.format(par['save_dir'],par['run_number'],i), vis)
