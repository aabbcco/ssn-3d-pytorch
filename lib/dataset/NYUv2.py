import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt


def convert_label(label):

    onehot = np.zeros(
        (1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


def convert_spix(label):

    onehot = np.zeros(
        (1, 200, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 200:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class NYUv2(Dataset):
    def __init__(self, root, split="train", color_transforms=None, geo_transforms=None):
        self.data_dir = os.path.join(root, split, 'nyu_images')
        self.gt_dir = os.path.join(root, split, 'nyu_labels')
        self.spix_dir = os.path.join(root, split, 'nyu_spixs')

        self.index = os.listdir(self.data_dir)
        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms

    def __getitem__(self, idx):
        data = rgb2lab(plt.imread(os.path.join(
            self.data_dir, self.index[idx])))
        data = data.astype(np.float64)
        label = np.loadtxt(os.path.join(
            self.gt_dir, self.index[idx][:-4] + '.csv'), dtype=np.int16, delimiter=',')
        spix = np.loadtxt(np.loadtxt(os.path.join(
            self.spix_dir, self.index[idx][:-4] + '.csv'), dtype=np.int16, delimiter=','))

        label = convert_label(label)
        label = torch.from_numpy(label)

        data = (torch.from_numpy(data)).permute(2, 0, 1)

        spix = convert_spix(spix)
        spix = torch.from_numpy(spix)

        return data, label, spix, (self.index[idx])[:-4]

    def __len__(self):
        return len(self.index)
