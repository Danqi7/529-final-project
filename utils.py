import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import numpy as np
import PIL


class DUTSDataset(Dataset):
    def __init__(self, root_dir, type, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.type = type
        self.image_dir = os.path.basename(root_dir) + '-Image'
        self.mask_dir = os.path.basename(root_dir) + '-Mask'

        image_files = os.listdir(os.path.join(root_dir, self.image_dir))
        mask_files = os.listdir(os.path.join(root_dir, self.mask_dir))

        image_and_mask = []
        # Mask: .png, Image: .jpg
        for i in range(len(image_files)):
            image_name = image_files[i]
            # 'ILSVRC2012_test_00004530.jpg' -> 'ILSVRC2012_test_00004530'
            image_id = image_name.split('.')[0]
            # 'ILSVRC2012_test_00004530.png'
            mask_name = '.'.join([image_id, 'png'])
            image_and_mask.append((image_name, mask_name))

        self.image_and_mask = image_and_mask

    def __len__(self):
        return len(self.image_and_mask)

    def __getitem__(self, index):
        image_name, mask_name = self.image_and_mask[index]
        image_name = os.path.join(self.root_dir, self.image_dir, image_name)
        mask_name = os.path.join(self.root_dir, self.mask_dir, mask_name)
        image = PIL.Image.open(image_name)
        mask = PIL.Image.open(mask_name)

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return (image, mask, image_name)

    def __str__(self):
        return 'data type: %s\nNumber of data: %d' % (self.type, len(self.image_and_mask))


def load_data():
    #data = datasets.VOCSegmentation('./data', "2012", image_set="train", download=True)
    unNormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                           std=[1/0.229, 1/0.224, 1/0.225]),
                                      transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                           std=[1., 1., 1.]),
                                      ])

    image_size = 224
    train_dataset = DUTSDataset(root_dir='./data/DUTS-TR',
                                type='train',
                                image_transform=transforms.Compose([
                                    transforms.Resize(
                                        (image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ]),
                                mask_transform=transforms.Compose([
                                    transforms.Resize(
                                        (image_size, image_size)),
                                    transforms.ToTensor(),
                                ]))

    test_dataset = DUTSDataset(root_dir='./data/DUTS-TE',
                               type='test',
                               image_transform=transforms.Compose([
                                    transforms.Resize(
                                        (image_size, image_size)),
                                    #transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ]),
                               mask_transform=transforms.Compose([
                                   transforms.Resize((image_size, image_size)),
                                   transforms.ToTensor(),
                               ]))

    print(train_dataset)
    print(test_dataset)

    # print(test_dataset)
    # print(test_dataset[0][0][:10])
    # print(test_dataset[0][1][:10])
    #print('img: ', test_dataset[0][0].shape)
    #print('mask: ', test_dataset[0][1].shape)

    # Visualize
    #sample = unNormalize(test_dataset[0][0]).numpy()
    # sample = test_dataset[0][0].numpy()
    # mask = test_dataset[0][1].numpy()
    # sample = np.transpose(sample, (1, 2, 0))
    # mask = np.transpose(mask, (1, 2, 0))
    # plt.imshow(sample)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()

    return train_dataset, test_dataset
