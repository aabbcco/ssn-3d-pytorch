from torch.utils.data import Dataset
from torch import Tensor
import os
import h5py
import numpy as np


def convert_label(label):

    onehot = np.zeros(
        (50, label.shape[0])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[ct, :] = (label == t)
        ct = ct + 1

    return onehot


class shapenet(Dataset):
    def __init__(self, datafolder, split='train'):
        '''
        datafolder: folder containing shapenet result
        split: options in train,test and val
        '''
        assert split in ['train', 'val', 'test'], "split not exist"
        self.listname = split + "_hdf5_file_list.txt"
        self.data = []
        self.label = []
        #the two method performs the same speed
        #the reason might be reinit of the memory
        # datas = []
        # labels = []
        datafile = open(os.path.join(datafolder, self.listname), 'r')
        flist = datafile.readlines()
        for fname in flist:
            #too slow,find a way to boost it
            #fucking GIL
            f = h5py.File(os.path.join(datafolder, fname.split('\n')[0]), 'r')
            self.data.extend(f['data'])
            self.label.extend(f['pid'])
            # datas.append(f['data'])
            # labels.append(f['pid'])
        # self.data = np.array(datas)
        # self.label = np.array(labels)

    def __getitem__(self, idx):
        return Tensor(self.data[idx]).transpose(1, 0), Tensor(convert_label(self.label[idx])), self.label[idx]

    def __len__(self):
        return len(self.data)
