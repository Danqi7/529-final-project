import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import numpy as np
import PIL

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

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
        image = pil_loader(image_name)
        mask = PIL.Image.open(mask_name)

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        # if image.shape[2] != 3 or mask.shape[2] != 1:

        return (image, mask, image_name)

    def __str__(self):
        return 'data type: %s\nNumber of data: %d' % (self.type, len(self.image_and_mask))


class HKUDataset(Dataset):
    def __init__(self, root_dir, type, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.type = type
        self.image_dir = os.path.basename(root_dir) + 'imgs'
        self.mask_dir = os.path.basename(root_dir) + 'gt'

        image_files = os.listdir(os.path.join(root_dir, self.image_dir))
        mask_files = os.listdir(os.path.join(root_dir, self.mask_dir))

        image_and_mask = []
        # Mask: .png, Image: .png
        for i in range(len(image_files)):
            #'0004.png'
            image_name = image_files[i]
            image_and_mask.append((image_name, image_name))

        self.image_and_mask = image_and_mask

    def __len__(self):
        return len(self.image_and_mask)

    def __getitem__(self, index):
        image_name, mask_name = self.image_and_mask[index]
        image_name = os.path.join(self.root_dir, self.image_dir, image_name)
        mask_name = os.path.join(self.root_dir, self.mask_dir, mask_name)
        image = pil_loader(image_name)
        mask = PIL.Image.open(mask_name)
        if len(np.array(image).shape) < 3:
            print('image.shape: ', np.array(image).shape)
            print('maks.shjape: ', np.array(mask).shape)
            print(image_name)

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        # if image.shape[2] != 3 or mask.shape[2] != 1:

        return (image, mask, image_name)

    def __str__(self):
        return 'data type: %s\nNumber of data: %d' % (self.type, len(self.image_and_mask))


def load_data(dataset_name):
    '''
        dataset_name: DUTS | HKU-IS |
    '''
    image_size = 224
    unNormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                           std=[1/0.229, 1/0.224, 1/0.225]),
                                      transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                           std=[1., 1., 1.]),
                                      ])

    if dataset_name == 'DUTS':
        train_data = DUTSDataset(root_dir='./data/DUTS-TR',
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
                                     transforms.Grayscale(
                                         num_output_channels=1),
                                     transforms.ToTensor(),
                                 ]))
        test_data = DUTSDataset(root_dir='./data/DUTS-TE',
                                type='test',
                                image_transform=transforms.Compose([
                                    transforms.Resize(
                                        (image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ]),
                                mask_transform=transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                ]))
    elif dataset_name == 'HKU':
        test_data = HKUDataset(root_dir="./HKU-IS/",
                            type="test",
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
                                transforms.Grayscale(
                                    num_output_channels=1),
                                transforms.ToTensor(),
                            ]))
        train_data = None

    print(train_data)
    print(test_data)

    # print(test_dataset)
    # print(test_dataset[0][0][:10])
    # print(test_dataset[0][1][:10])
    #print('img: ', test_dataset[0][0].shape)
    #print('mask: ', test_dataset[0][1].shape)

    # Visualize

    # sample = unNormalize(train_data[1][0]).numpy()
    # #sample = test_dataset[0][0].numpy()
    # mask = train_data[1][1].numpy()
    # sample = np.transpose(sample, (1, 2, 0))
    # mask = np.transpose(mask, (1, 2, 0))
    # plt.imshow(sample)
    # plt.show()
    # plt.imshow(mask.squeeze())
    # plt.show()

    return train_data, test_data


if __name__ == "__main__":
    unNormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                           std=[1/0.229, 1/0.224, 1/0.225]),
                                      transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                           std=[1., 1., 1.]),
                                      ])

    image_size = 224
    #TODO: normaliztion image mean,std might need to be computed
    DUTS_data = DUTSDataset(root_dir="./data/DUTS-TE",
                          type="test",
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
                              transforms.Grayscale(
                                  num_output_channels=1),
                              transforms.ToTensor(),
                          ]))
    HKU_data = HKUDataset(root_dir="./HKU-IS/",
                          type="test",
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
                              transforms.Grayscale(
                                  num_output_channels=1),
                              transforms.ToTensor(),
                          ]))
    # HKU_data = HKUDataset(root_dir="./HKU-IS/",
    #                       type="test",
    #                       image_transform=transforms.Compose([
    #                           transforms.Resize(
    #                               (image_size, image_size)),
    #                           transforms.ToTensor(),
    #                       ]),
    #                       mask_transform=transforms.Compose([
    #                           transforms.Resize(
    #                               (image_size, image_size)),
    #                           transforms.Grayscale(
    #                               num_output_channels=1),
    #                           transforms.ToTensor(),
    #                       ]))
    
    print(HKU_data)
    print(DUTS_data)

    data_loader = torch.utils.data.DataLoader(
        HKU_data, batch_size=16, shuffle=False)
    # DUTS_dataloader = torch.utils.data.DataLoader(
    #     DUTS_data, batch_size=len(DUTS_data), shuffle=False)
    
    
    data = next(iter(data_loader))
    # print(torch.mean(data[0],dim=(0,2,3)), torch.std(data[0], dim=(0,2,3)))

    #DUTS_test = next(iter(DUTS_dataloader))
    #print(torch.mean(DUTS_test[0], dim=(0, 2, 3)), torch.std(DUTS_test[0], dim=(0,2,3)))

    # Visualize
    print('data.shape: ', data[0][1].shape)
    print('data[0][2]', data[0][2].shape)
    sample = unNormalize(data[0][1]).numpy()
    #sample = test_dataset[0][0].numpy()
    mask = data[1][1].numpy()
    sample = np.transpose(sample, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    plt.imshow(sample)
    plt.show()
    plt.imshow(mask.squeeze())
    plt.show()
