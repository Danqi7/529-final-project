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

from utils import plot_and_save, save_model_info
from positional_embeddings import gaussian_pos_embedding
from data_utils import load_data

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
image_c = 3
GAUSSIAN_SIGMA = 90

'''
    positional_encoding: bool determines whether to add positional encodings
    If positional_encoding is True, VGG will be modified at pos_inject_layer
    and the input image will be augmented to have extra channel

    FCN8s: residual connection up to w/8, h/8 (3rd layer)
'''
class FCN8s(nn.Module):
    def __init__(self,
                 n_class, pretrained=True, pretrained_model='vgg16',
                 positional_encoding=False, pos_embed_type="Random",
                 pos_inject_layer=0, pos_inject_side="encoder"):
        super().__init__()
        self.n_class = n_class
        self.positional_encoding = positional_encoding
        self.pos_embed_type = pos_embed_type

        # Encoder
        if pretrained_model == 'vgg16':
            pretrained_net = VGGNet(pretrained=pretrained, requires_grad=True,
                                    positional_encoding=positional_encoding, 
                                    pos_inject_layer=pos_inject_layer,
                                    show_params=False, show_params_values=False)
            self.pretrained_net = pretrained_net
        elif pretrained_model ==  'vgg11':
            pretrained_net = VGGNet(pretrained=pretrained, model='vgg11', requires_grad=True,
                                    positional_encoding=positional_encoding,
                                    pos_inject_layer=pos_inject_layer,
                                    show_params=False, show_params_values=False)
            self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        # Decoder
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
        # x : [b x 3 x h x w]
        batch_size = x.shape[0]
        image_size = x.shape[2]
        if self.positional_encoding:
            pos_embed = None  # [img_size x img_size]
            if self.pos_embed_type == 'Gaussian':
                pos_embed = torch.from_numpy(
                    gaussian_pos_embedding(image_size, sigma=90))
                pos_embed = pos_embed.type(torch.FloatTensor)
            elif self.pos_embed_type == 'Random':
                pos_embed = torch.rand(
                    (image_size, image_size))  # random embed
            pos_embed = torch.unsqueeze(pos_embed.repeat(
                (batch_size, 1, 1)), dim=1)  # [b x 1 x h x w]
            pos_embed = pos_embed.to(device)
            x = torch.cat((x, pos_embed), dim=1)  # [b x 4 x h x w]
            #print('after adding pos embed x.shape: ', x.shape)

        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        #print('forwar encoder x5.shape: ', x5.shape)

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

'''
    positional_encoding: bool determines whether to add positional encodings
    If positional_encoding is True, VGG will be modified at pos_inject_layer
    and the input image will be augmented to have extra channel
'''
class FCN32s(nn.Module):
    def __init__(self,
                 n_class, pretrained=True, pretrained_model=None,
                 positional_encoding=False, pos_embed_type="Random",
                 pos_inject_layer=0, pos_inject_side="encoder"):
        super().__init__()
        self.n_class = n_class
        self.image_size = 224
        self.batch_size = 22
        self.positional_encoding = positional_encoding
        self.pos_embed_type = pos_embed_type
        
        # Encoder
        if pretrained_model == 'vgg16':
            pretrained_net = VGGNet(pretrained=pretrained, requires_grad=True,
                                    positional_encoding=positional_encoding, 
                                    pos_inject_layer=pos_inject_layer,
                                    show_params=False, show_params_values=False)
            self.pretrained_net = pretrained_net
        elif pretrained_model ==  'vgg11':
            pretrained_net = VGGNet(pretrained=pretrained, model='vgg11', requires_grad=True,
                                    positional_encoding=positional_encoding,
                                    pos_inject_layer=pos_inject_layer,
                                    show_params=False, show_params_values=False)
            self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        # self.deconv1 = nn.ConvTranspose2d(
        #     512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(512)
        # self.deconv2 = nn.ConvTranspose2d(
        #     512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.deconv3 = nn.ConvTranspose2d(
        #     256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.deconv4 = nn.ConvTranspose2d(
        #     128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.deconv5 = nn.ConvTranspose2d(
        #     64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn5 = nn.BatchNorm2d(32)
        # self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        # Positional Encoding
        if self.positional_encoding:
            pos_embed = None  # [img_size x img_size]
            if self.pos_embed_type == 'Gaussian':
                pos_embed = torch.from_numpy(
                    gaussian_pos_embedding(self.image_size, sigma=90))
                pos_embed = pos_embed.type(torch.FloatTensor)
            elif self.pos_embed_type == 'Random':
                pos_embed = torch.rand(
                    (self.image_size, self.image_size))  # random embed
            pos_embed = torch.unsqueeze(pos_embed.repeat(
                (self.batch_size, 1, 1)), dim=1)  # [b x 1 x h x w]
            pos_embed = pos_embed.to(device)
            self.pos_embed = pos_embed


    def forward(self, x):
        # x : [b x 3 x h x w]
        batch_size = x.shape[0]
        image_size = x.shape[2]
        if self.positional_encoding:
            # pos_embed = None  # [img_size x img_size]
            # if self.pos_embed_type == 'Gaussian':
            #     pos_embed = torch.from_numpy(gaussian_pos_embedding(image_size, sigma=90))
            #     pos_embed = pos_embed.type(torch.FloatTensor)
            # elif self.pos_embed_type == 'Random':
            #     pos_embed = torch.rand((image_size, image_size))  # random embed
            # pos_embed = torch.unsqueeze(pos_embed.repeat((batch_size, 1, 1)), dim=1)  # [b x 1 x h x w]
            # pos_embed = pos_embed.to(device)
            x = torch.cat((x, self.pos_embed), dim=1)  # [b x 4 x h x w]
            #print('after adding pos embed x.shape: ', x.shape)

        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        #print('forwar encoder x5.shape: ', x5.shape)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(self.relu(self.deconv1(x5)))
        #print('score.shape: ', score.shape)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        #print('score.shape: ', score.shape)
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        #print('score.shape: ', score.shape)
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        #print('score.shape: ', score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        
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

'''
    MVP for intializing encoder to have extra positional channels
'''

def make_layers(cfg, batch_norm=False, positional_encoding=False) -> nn.Sequential:
    layers = []
    if positional_encoding:
        in_channels = 4
    else:
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
    #print('layers~~~~~~: ', len(layers), layers)
    return nn.Sequential(*layers)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, positional_encoding=False, pos_inject_layer=0, show_params=False, show_params_values=False):
        super().__init__(make_layers(cfg[model], positional_encoding=positional_encoding))
        self.ranges = ranges[model]
        
        model_dict = self.state_dict()
        #print('model_dict: ', model_dict.keys())

        if pretrained:
            if model == 'vgg16':                
                # Load pretrained vgg weights on ImageNet 
                pretrained_dict = models.vgg16(pretrained=True, progress=True).state_dict()
                #print('pretrained_dict: ')
                print(pretrained_dict.keys())
            elif model == 'vgg11':
                pretrained_dict = models.vgg11(
                    pretrained=True, progress=True).state_dict()
                #print('pretrained_dict: ')
                print(pretrained_dict.keys())

            # Load pretrained weights but leave the pos_embed weights initialized
            if positional_encoding:
                for key, value in pretrained_dict.items():
                    if key == 'features.%d.weight'%(pos_inject_layer):
                        print('hi')
                        #print('value.shape: ', value.shape)
                        #print('value: ', value)
                        model_dict[key][:, 1:, :, :] = value # [out_channel x in_channel x 3 x 3]
                    else:
                        model_dict[key] = value
            else: # No modification on VGG, directly load
                model_dict = pretrained_dict
            
            self.load_state_dict(model_dict)
            #exec("self.load_state_dict(models.%s(pretrained=True, progress=True).state_dict(), strict=False)" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size(), param.requires_grad)

        if show_params_values:
            named_param = next(self.named_parameters())
            name, param = named_param
            #print(name, param.data[0, 0, :, :])
            #print('pretrained weights: ', param.data[0, 1, :, :])

    def forward(self, x, positional_encoding=False):
        # x : [b x 3 x h x w] or [b x 4 x h x w]
        output = {}
        
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        #print('self.features: ', self.features)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x

        return output
