import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import json
import numpy as np
import PIL

def plot_and_save(pts, name, save_file, ptype, freq=10):
    plt.figure()
    plt.title('%s %s per %d iterations'%(ptype, name, freq))
    plt.plot(pts)
    plt.xlabel('ith ' + str(freq) + ' iterations')
    plt.ylabel(name)
    plt.savefig(save_file + '/%s_%s.png'%(ptype, name))


def save_PRCurve_fmeasures(plot_results, save_file):
    '''
        plot_results: Dict of storing pr, rc, fmeasures for each datasets
    '''
    with open(os.path.join(save_file, 'plot_results.txt'), 'w') as fh:
        json.dump(plot_results, fh)


def save_model_info(model, results, all_results, average_result, params, elapsed_time, save_file):
    losses, precision, recall, fmeasure, mae = results
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    f = open(save_file + "/model_info.txt", "a")
    content = "model: " + save_file + "/model.ckpt\n" + "\nAverage Loss: " + str(np.mean(losses))
    content += "\nprecision: " + str(precision) + "\nrecall: " + str(recall)
    content += "\nAll Datasets Eval Results: " + all_results.__str__()
    content += "\nAverage Result across all test datasets: " + average_result.__str__()
    content += "\nlr: " + str(learning_rate) + "\nbatch size: " + str(batch_size) + "\nnum_epochs: " + str(num_epochs)
    content += "\nTraining Time: " + str(elapsed_time)
    content += "\nArchitecture: " + model.__str__()

    f.write(content)
    f.close()
