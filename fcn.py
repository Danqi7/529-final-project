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
import numpy as np
import PIL

from utils import load_data

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
num_epochs = 1
num_classes = 50
batch_size = 16
image_size = 128
image_c = 3

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


def train(model, optim, loss_function, train_loader, params):
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
    best_model = model
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

            # Eval on full testing on the end batch of epoch
            # if (e % num_epoch_per_eval == 0 and i == num_iters_per_epoch - 1) or (e == num_epochs-1 and i == num_iters_per_epoch - 1):
            #     print(
            #         "[%d/%d][%d/%d]\tEvaluating..."
            #         % (
            #             e,
            #             num_epochs,
            #             i,
            #             len(train_loader)
            #         )
            #     )
            #     class_acc, attr_acc, confusion_M = eval_model.eval(model)
            #     class_accs.append(class_acc)
            #     attr_accs.append(attr_acc)

            #     # Save best model
            #     if class_acc > best_acc:
            #       best_conf_mat = confusion_M
            #       best_attr_acc = attr_acc
            #       best_acc = class_acc
            #       best_model = model

    #torch.save(best_model, save_file + "/model.ckpt")
    #wandb.log({'losses': losses, 'class_accs': class_accs, "attr_accs": attr_accs})
    return losses

if __name__ == "__main__":
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    batch_size, n_class, h, w = 1, 1, 224, 224

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
    assert output.size() == torch.Size([batch_size, n_class, h, w])
    
    print('Check Pass')


    fcn_model = fcn_model.to(device)
    num_iters_per_print = 5
    num_epochs = num_epochs
    num_epoch_per_eval = 10
    save_file = ''
    params = {
        "batch_size": batch_size,
        "num_iters_per_print": num_iters_per_print,
        "num_epoch_per_eval": num_epoch_per_eval,
        "num_epochs": num_epochs,
        "save_file": save_file
    }
    learning_rate = 1e-4
    optim = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # Train
    losses = train(fcn_model, optim, loss_function, train_loader, params)

