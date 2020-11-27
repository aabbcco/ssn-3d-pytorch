import math
import numpy as np
import torch
import os

from skimage.color import rgb2lab
from lib.dataset.shapenet import shapenet
from lib.utils.pointcloud_io import write
from torch.utils.data import DataLoader
from lib.ssn.ssn import soft_slic_all


@torch.no_grad()
def inference(pointcloud, nspix, n_iter, fdim=None, pos_scale=10, weight=None):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from model_ptnet import PointNet_SSN
        model = PointNet_SSN(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        def model(data): return soft_slic_all(data, data[:, :, :nspix], n_iter)

    print(model)

    inputs = pos_scale * pointcloud
    inputs = inputs.to("cuda")

    Q, H, center, feature = model(inputs)

    labels = H.to("cpu").detach().numpy()

    return labels, center


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default='log/model-shapenet.pth',
                        type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=10, type=int,
                        help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", default=10, type=float)
    args = parser.parse_args()

    data = shapenet("../shapenet_part_seg_hdf5_data", split='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)

    pointcloud, label = iter(loader).next()
    s = time.time()
    label, center = inference(pointcloud, args.nspix, args.niter,
                              args.fdim, args.pos_scale, args.weight)
    print(f"time {time.time() - s}sec")
    ptcloud = pointcloud.squeeze(0).permute(1, 0).numpy()
    ptcloud = np.concatenate((ptcloud, label.transpose(1, 0)), axis=-1)
    write.tobcd(ptcloud, 'xyzl', 'shapenet_pred.pcd')
