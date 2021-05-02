import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#from torchvision.models.vgg import VGG
from vgg import VGG

import matplotlib.pyplot as plt

import os
import time
import numpy as np
import argparse
import PIL

# from utils import plot_and_save, save_model_info
from positional_embeddings import gaussian_pos_embedding
from fcn import FCN32s, FCN8s

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Hyperparameters
image_size = 224
image_c = 3
n_class = 1
GAUSSIAN_SIGMA = 90

def eval_sample(model, test_data, params):
    '''
    Evaluate SOD model on test data
    If prob > threshold, then 1

    Output:
        precision: tp / tp + fp
        recall: tp / tp + fn
        f_measure: 
        MAE: 
    '''
    tp = 0
    fp = 0
    fn = 0
    batch_mae = []
    batch_size = []

    threshold = 0.5
    belta_sq = 0.3

    model.eval()  # set model to eval mode for bn, dropout behave properly
    with torch.no_grad():
        x = test_data[0].to(device)  # b x C x W x H
        y = test_data[1].to(device)  # b x 1 x W x H

        output = model(x).cpu().numpy()  # b x 1 x W x H
        pred_mask = np.copy(output)
        y = y.cpu().numpy()
        pred_mask_round = np.round(pred_mask)
        # pred_mask[output > threshold] = 1
        # pred_mask[output <= threshold] = 0

        tp += np.sum(pred_mask_round[y == 1] == 1)
        fp += np.sum(pred_mask_round[y == 0] == 1)
        fn += np.sum(pred_mask_round[y == 1] == 0)

        mae = np.mean(np.abs(pred_mask-y))
        batch_mae.append(mae)
        batch_size.append(x.shape[0])


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq**2) * precision * recall / \
        ((belta_sq**2) * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)
    print(
        "precision:%.2f\t recall:%.2f\t f-measure:%.2f\t MAE:%.2f\t"
        % (
            precision,
            recall,
            f_measure,
            MAE
        )
    )

    return precision, recall, f_measure, MAE


def eval(model, test_loader, params):
    '''
    Evaluate SOD model on test data
    If prob > threshold, then 1

    Output:
        precision: tp / tp + fp
        recall: tp / tp + fn
        f_measure: 
        MAE: 
    '''
    tp = 0
    fp = 0
    fn = 0
    batch_mae = []
    batch_size = []

    threshold = 0.5
    belta_sq = 0.3

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x = data[0].to(device)  # b x C x W x H
            y = data[1].to(device)  # b x 1 x W x H

            output = model(x).cpu().numpy()  # b x 1 x W x H
            pred_mask = np.copy(output)
            y = y.cpu().numpy()
            pred_mask_round = np.round(pred_mask)
            # pred_mask[output > threshold] = 1
            # pred_mask[output <= threshold] = 0

            tp += np.sum(pred_mask_round[y == 1] == 1)
            fp += np.sum(pred_mask_round[y == 0] == 1)
            fn += np.sum(pred_mask_round[y == 1] == 0)

            mae = np.mean(np.abs(pred_mask-y))
            batch_mae.append(mae)
            batch_size.append(x.shape[0])

            # if i % 100 == 0:
            #     print(
            #         "[%d/%d]\ttp:%d\t fp:%d\t fn:%d\t mae:%f\t"
            #         % (
            #             i,
            #             len(test_loader),
            #             tp,
            #             fp,
            #             fn,
            #             mae
            #         )
            #     )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq**2) * precision * recall / \
        ((belta_sq**2) * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)

    return precision, recall, f_measure, MAE
