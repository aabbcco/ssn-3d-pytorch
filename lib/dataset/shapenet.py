from torch.utils.data import Dataset
from torch import Tensor
import os
import h5py
import numpy as np


def getFiles_full(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]


class shapenet(Dataset):
    """
    shapenet dataset for Pytorch Dataloader\n
    too slow in init time,need another implemention

    Args:
        datafolder (string): folder containing shapenet result
        split (str, optional): options in train,test and val. Defaults to 'train'.
"""

    def __init__(self, datafolder, split='train'):
        assert split in ['train', 'val', 'test'], "split not exist"
        listname = split + "_hdf5_file_list.txt"
        datalist = []
        labelist = []
        datafile = open(os.path.join(datafolder, listname), 'r')
        flist = datafile.readlines()
        for fname in flist:
            f = h5py.File(os.path.join(datafolder, fname.split('\n')[0]), 'r')
            datalist.append(np.array(f['data']))
            labelist.append(np.array(f['pid']))
        self.data = np.concatenate(datalist, axis=0).transpose(0, 2, 1)
        self.label = np.concatenate(labelist, axis=0)

    def __getitem__(self, idx):
        return Tensor(self.data[idx]), Tensor(self.label[idx])

    def __len__(self):
        return len(self.data)


class shapenet_spix(Dataset):
    def __init__(self, datafolder, split='train'):
        assert split in ['train', 'val', 'test'], "split not exist"
        filepath = os.path.join(datafolder, split)
        datalist = []
        labelist = []
        spixlist = []
        flist = getFiles_full(filepath, '.h5')
        for fname in flist:
            f = h5py.File(os.path.join(datafolder, fname.split('\n')[0]), 'r')
            datalist.append(np.array(f['data']))
            labelist.append(np.array(f['label']))
            spixlist.append(np.array(f['spix']))
        self.data = np.concatenate(datalist,axis=0).transpose(0,2,1)
        self.label = np.concatenate(labelist, axis=0)
        self.spix = np.concatenate(spixlist, axis=0)

    def __getitem__(self, idx):
        return Tensor(self.data[idx]), Tensor(self.label[idx]), Tensor(self.spix[idx])

    def __len__(self):
        return len(self.data)
