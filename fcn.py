import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

import matplotlib.pyplot as plt

import os
import time
import numpy as np
import argparse
import PIL

from utils import load_data, plot_and_save, save_model_info

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
num_epochs = 1
num_classes = 50
batch_size = 16
image_size = 128
image_c = 3


class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv2(score))
        # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        sigmoid = nn.Sigmoid()
        prob = sigmoid(score)

        return prob # size=(N, n_class, x.H/1, x.W/1)

class FCN32s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        sigmoid = nn.Sigmoid()
        prob = sigmoid(score)

        return prob  # size=(N, n_class, x.H/1, x.W/1)


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        #print('self.features: ', self.features)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x

        return output


def train(model, optim, loss_function, train_loader, sample_test, params, test_params):
    # Params
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    num_iters_per_print = params['num_iters_per_print']
    num_epoch_per_eval = params['num_epoch_per_eval']
    save_file = params['save_file']

    # Print some info about train data
    num_data = len(train_loader) * batch_size
    num_iters_per_epoch = num_data / batch_size
    print('Total number of Iterations: ', num_iters_per_epoch * num_epochs)
    print("num_data: ", num_data, "\nnum_iters_per_epoch: ", num_iters_per_epoch)

    losses = []
    precisions = []
    recalls = []
    maes = []
    fmeasures = []

    best_mae = 1
    model.train() # Set model to train mode
    for e in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            x = data[0].to(device)  # b x C x W x H
            y = data[1].to(device) # b x 1 x W x H

            # Forward & backward pass
            model.zero_grad()

            output = model(x)  # b x 1 x W x H
            loss = loss_function(output, y)

            loss.backward()
            optim.step()

            # Bookkeeping
            if i % num_iters_per_print == 0 or i == num_iters_per_epoch-1:
              print(
                  "[%d/%d][%d/%d]\tLoss_D: %.4f\t"
                  % (
                      e,
                      num_epochs,
                      i,
                      len(train_loader),
                      loss
                  )
              )
              losses.append(loss.item())

            # Eval on sample testing on the end batch of epoch
            if (e % num_epoch_per_eval == 0 and i == num_iters_per_epoch - 1) or (e == num_epochs-1 and i == num_iters_per_epoch - 1):
                print(
                    "[%d/%d][%d/%d]\tEvaluating..."
                    % (
                        e,
                        num_epochs,
                        i,
                        len(train_loader)
                    )
                )
                pr, rc, fm, mae = eval_sample(model, sample_test, test_params)
                precisions.append(pr)
                recalls.append(rc)
                fmeasures.append(fm)
                maes.append(mae)
                
                # Save best model
                if mae > best_mae and save_file != "":
                    best_mae = mae
                    torch.save(model, save_file + "/model.ckpt")

    #wandb.log({'losses': losses, 'class_accs': class_accs, "attr_accs": attr_accs})
    return losses, precisions, recalls, fmeasures, maes

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

    model.eval() # set model to eval mode for bn, dropout behave properly
    with torch.no_grad():
        x = test_data[0].to(device)  # b x C x W x H
        y = test_data[1].to(device)  # b x 1 x W x H

        output = model(x).cpu().numpy()  # b x 1 x W x H
        pred_mask = np.copy(output)
        y = y.cpu().numpy()
        pred_mask[output > threshold] = 1
        pred_mask[output <= threshold] = 0

        tp += np.sum(pred_mask[y == 1] == 1)
        fp += np.sum(pred_mask[y == 0] == 1)
        fn += np.sum(pred_mask[y == 1] == 0)

        mae = np.mean(np.abs(pred_mask-y))
        batch_mae.append(mae)
        batch_size.append(x.shape[0])

        print(
            "tp:%d\t fp:%d\t fn:%d\t mae:%f\t"
            % (
                tp,
                fp,
                fn,
                mae
            )
        )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq) * precision * recall / \
        (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)

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
            pred_mask[output > threshold] = 1
            pred_mask[output <= threshold] = 0

            tp += np.sum(pred_mask[y==1]==1)
            fp += np.sum(pred_mask[y==0]==1)
            fn += np.sum(pred_mask[y==1]==0)
            
            mae = np.mean(np.abs(pred_mask-y))
            batch_mae.append(mae)
            batch_size.append(x.shape[0])

            print(
                "[%d/%d]\ttp:%d\t fp:%d\t fn:%d\t mae:%f\t"
                % (
                    i,
                    len(test_loader),
                    tp,
                    fp,
                    fn,
                    mae
                )
            )
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq) * precision * recall / (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / np.sum(batch_size)

    return precision, recall, f_measure, MAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_files", type=str, required=True,
                        help="Where to store the trained model")
    parser.add_argument("--batch_size", default=16,
                        type=int, help="How many WordPiece tokens to use")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="How many WordPiece tokens to use")

    args = parser.parse_args()

    if not os.path.exists(args.store_files):
        os.makedirs(args.store_files)


    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    n_class, h, w = 1, 224, 224

    # test output size
    vgg_model = VGGNet(requires_grad=True)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))
    # output = vgg_model(input)
    # assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])

    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    #input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    input = next(iter(train_data))[0].reshape((1,3,224,224))
    print('input.shape: ', input.shape)
    output = fcn_model(input)
    assert output.size() == torch.Size([1, n_class, h, w])
    print('Check Pass')


    fcn_model = fcn_model.to(device)
    num_iters_per_print = 5
    num_epochs = num_epochs
    num_iters_per_eval = 10
    save_file = ''
    params = {
        "batch_size": args.batch_size,
        "num_iters_per_print": num_iters_per_print,
        "num_epoch_per_eval": num_iters_per_eval,
        "num_epochs": args.num_epochs,
        "save_file": args.store_files
    }
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5
    }
    learning_rate = 1e-4
    optim = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # Train
    print(fcn_model)
    validation_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True)
    sample_validation = next(iter(validation_loader))
    start_time = time.time()
    losses, precisions, recalls, fmeasures, maes = train(
        fcn_model, optim, loss_function, train_loader, sample_validation, params, eval_params)

    elapsed_time = time.time() - start_time
    print('Training Finished in : ', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))

    # Evaluate
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5
    }
    pr, rc, fm, mae = eval(fcn_model, test_loader, params)

    results = (np.mean(losses), pr, rc, fm, mae)
    print(pr, rc, fm, mae)

    # Plot
    plot_and_save(losses, "loss", args.store_files,
                  "train", freq=num_iters_per_print)
    plot_and_save(precisions, "precision", args.store_files,
                  "validation", freq=num_iters_per_eval)
    plot_and_save(recalls, "recall", args.store_files,
                  "validation", freq=num_iters_per_eval)
    plot_and_save(fmeasures, "f-measure", args.store_files,
                  "validation", freq=num_iters_per_eval)
    plot_and_save(maes, "MAE", args.store_files,
                  "validation", freq=num_iters_per_eval)

    # Save model info
    save_model_info(model, results, params, elapsed_time, args.store_files)

    # Visualize
    


