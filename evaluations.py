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
from matplotlib import cm

import os
import time
import numpy as np
import argparse
import PIL
import json

# from utils import plot_and_save, save_model_info
from positional_embeddings import gaussian_pos_embedding
from fcn import FCN32s, FCN8s
from data_utils import load_data

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Hyperparameters
image_size = 224
image_c = 3
n_class = 1
GAUSSIAN_SIGMA = 90

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

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

    print('tp, fp, fn: ', tp, fp, fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq) * precision * recall / \
        (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)
    print(
        "precision:%.4f\t recall:%.4f\t f-measure:%.4f\t MAE:%.4f\t"
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

    nthresh = 99
    tps = np.zeros([nthresh])
    fps = np.zeros([nthresh])
    fns = np.zeros([nthresh])

    stepsize = 1. / (nthresh + 1)
    threshs = np.zeros([nthresh])

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

            # PR Curve
            for i in range(nthresh):
                thresh = (i + 1) * stepsize
                threshs[i] = thresh
                tps[i] += np.sum(pred_mask[y == 1] >= thresh)
                fps[i] += np.sum(pred_mask[y == 0] >= thresh)
                fns[i] += np.sum(pred_mask[y == 1] < thresh)

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
    f_measure = (1+belta_sq) * precision * recall / \
        (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)

    # Precisions, Recalls, Curve
    precisions = np.array(tps) / (np.array(tps) + np.array(fps))
    recalls = np.array(tps) / (np.array(tps) + np.array(fns))
    f_measures = (1+belta_sq) * precisions * recalls / \
        (belta_sq * precisions + recalls)
    fmax = np.max(f_measures)
    fmax_thresh = (np.argmax(f_measures) + 1) * stepsize
    print('fmax: ', fmax, '\t at threshold: ', fmax_thresh)
    plot_result = (list(precisions), list(recalls), list(f_measures))

    return precision, recall, f_measure, MAE, fmax, fmax_thresh, plot_result

#TODO: precision/recal curve + per image adjustable thresholding + f-measure / threshold curv
def precision_recall(model, test_loader, params):
    model.eval()
    
    nthresh = 99
    tps = np.zeros([nthresh])
    fps = np.zeros([nthresh])
    fns = np.zeros([nthresh])

    stepsize = 1. / (nthresh + 1)
    threshs = np.zeros([nthresh])
    
    mae_list = []   # List to save mean absolute error of each image
    belta_sq = 0.3
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x = data[0].to(device)  # b x C x W x H
            print('x.shape', x.shape)
            y = data[1].to(device)  # b x 1 x W x H

            output = model(x).cpu().numpy()  # b x 1 x W x H
            pred_mask = np.copy(output)
            y = y.cpu().numpy()

            for i in range(nthresh):
                thresh = (i + 1) * stepsize
                threshs[i] = thresh
                tps[i] += np.sum(pred_mask[y == 1] >= thresh)
                fps[i] += np.sum(pred_mask[y == 0] >= thresh)
                fns[i] += np.sum(pred_mask[y == 1] < thresh)
    
    # precisions = np.zeros([nthresh])
    # recallss = np.zeros([nthresh])
    precisions = np.array(tps) / (np.array(tps) + np.array(fps))
    recalls = np.array(tps) / (np.array(tps) + np.array(fns))

    f_measures = (1+belta_sq) * precisions * recalls / \
        (belta_sq * precisions + recalls)
    
    fmax = np.max(f_measures)
    fmax_thresh = (np.argmax(f_measures) + 1) * stepsize
    print('fmax: ', fmax, '\t at threshold: ', fmax_thresh)

    #return precisions, recalls
    print(threshs)
    plt.figure()
    plt.title('Precision vs Recall')
    plt.plot(precisions, recalls)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()

    plt.figure()
    plt.title('F-measures')
    plt.plot(threshs, f_measures)
    plt.xlabel('threshs')
    plt.ylabel('F-measure')
    plt.show()

def visualize_mask(model, img_path, gt_path, save_file):
    '''
        Given SoD model, make mask prediction and visualize it along 
        the input img
    '''
    image_transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1)])

    img = pil_loader(img_path)
    gt_mask = PIL.Image.open(gt_path)
    img = image_transform(img)
    gt_mask = mask_transform(gt_mask)
    #print('gt_mask.shape: ', gt_mask.shape)
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img).to(device)
    #print(img.shape)
    pred_mask = model(img).detach().cpu().numpy()

    #print(pred_mask.shape)

    # Visualize
    unNormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                           std=[1/0.229, 1/0.224, 1/0.225]),
                                      transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                           std=[1., 1., 1.]),
                                      ])
    unNormImg = unNormalize(img)
    #print('~~~!!!====: ', unNormImg.shape)
    unNormImg = torch.squeeze(unNormImg.permute(2, 3, 1, 0), dim=3).cpu().numpy()
    #print('~~~!!!====: ', unNormImg.shape)
    pred_mask = np.squeeze(np.transpose(pred_mask, (2, 3, 1, 0)), axis=3)
    img_name = "".join(img_path.split('/')[1:])
    #print('pred_mask.shap: ', pred_mask.shape)
    #print(save_file + '/'+img_name+'_img')
    plt.imsave(save_file + '/'+img_name+'_img.jpg', unNormImg)
    plt.imsave(save_file + '/'+img_name+'_pred.png', np.squeeze(pred_mask,axis=2), cmap=cm.gray)
    gt_mask.save(save_file + '/'+img_name+'_gt.png')


def visualize(model, save_file):
     img_path = './ECSSD/images/0001.jpg'
     gt_path = './ECSSD/ground_truth_mask/0001.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/9.jpg'
     gt_path = './salObj/masks/9.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/25.jpg'
     gt_path = './salObj/masks/25.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/90.jpg'
     gt_path = './salObj/masks/90.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/118.jpg'
     gt_path = './salObj/masks/118.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/171.jpg'
     gt_path = './salObj/masks/171.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/170.jpg'
     gt_path = './salObj/masks/170.png'
     visualize_mask(model, img_path, gt_path, save_file)

     img_path = './salObj/imgs/458.jpg'
     gt_path = './salObj/masks/458.png'
     visualize_mask(model, img_path, gt_path, save_file)

if __name__ == "__main__":
#     model_path = './models/'
   
    # Load Model
    model_path = './models/pos_encoding0_injectlayer0_typeGaussian_encoder1619773570.4/fcn.pt'
    save_file = './models/pos_encoding0_injectlayer0_typeGaussian_encoder1619773570.4'
    model = FCN8s(1,False, 'vgg16', 3, True, False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
    
    visualize(model, save_file)
    
    # Test Data
    # batch_size = 16
    # _, PSCALS_test_data = load_data('PASCALS')
    # PSCALS_test_loader = torch.utils.data.DataLoader(
    #     PSCALS_test_data, batch_size=batch_size, shuffle=False)

    # precision_recall(model, PSCALS_test_loader, {})
