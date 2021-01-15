# External Modules
import torch
from torch import cuda, FloatTensor, LongTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import Union
from time import time
#comes from PointCNN.Pytorch repository
#https://github.com/hxdengBerkeley/PointCNN.Pytorch.git

# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]


def knn_indices_func_cpu(rep_pts: FloatTensor,  # (N, pts, dim)
                         pts: FloatTensor,      # (N, x, dim)
                         K: int, D=1
                         ) -> LongTensor:         # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    #time1 = time()
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(
            D*K + 1, algorithm="ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:, 1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis=0))
    #print("using cpu,time:{}s".format(time()-time1))
    return region_idx


def knn_indices_func_gpu(seed: cuda.FloatTensor,  # (B,C,npoint)
                         pts: cuda.FloatTensor,  # (B,C,N)
                         k: int
                         ) -> cuda.LongTensor:  # (N,npoint,K)
    """knn indices func reimplemented

<<<<<<< HEAD
    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref)**2, 2).squeeze()
        _, inds = torch.topk(dist2, k*d + 1, dim=1, largest=False)
        region_idx.append(inds[:, 1::d])
        print(inds[:,1::d].shape)
=======
    Args:
        seed    (cuda.FloatTensor)  : clusting seed->(B,C,npoint)
        pts     (cuda.FloatTensor)  : pointcloud using clusting method->(B,C,N) 
        l       (int)               : k neibor in knn 
    Returns:
        cuda.LongTensor: knn idx(B,npoint,k)
    """    
    _, _, N = seed.shape
    _, _, M = pts.shape
    mseed = seed.unsqueeze(-2).expand(-1, -1, M, -1)
    mpts = pts.unsqueeze(-1).expand(-1, -1, -1, N)
    mdist = torch.sum((mpts-mseed)**2, dim=1)
    _, idx = torch.topk(mdist, k=k+1, largest=False)
>>>>>>> 45cd06e41eba257ee1062b4745986fcd546b7718

    return idx[:, :, 1:]
