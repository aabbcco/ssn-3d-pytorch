import numpy as np
import torch
import os

import sys
sys.path.append(os.path.dirname("../"))

from lib.dataset.shapenet import shapenet_man
from lib.utils.pointcloud_io import write
from torch.utils.data import DataLoader
from models.model_ptnet import PointNet_SSNx_old
from lib.ssn.ssn import soft_slic_pknn


@torch.no_grad()
def inference(pointcloud, pos_scale=10, weight=None):
    """generate 3d spix

    Args:
        pointcloud (Tensor): Tensor of input pointcloud
        pos_scale (int, optional): coordinate multpilter. Defaults to 10.
        weight ([type], optional): model itself. Defaults to None.

    Returns:
        [type]: [description]
    """
    if weight is not None:
        model = weight
    else:
        raise Exception('model not loaded')
    inputs = pos_scale * pointcloud
    inputs = inputs.to("cuda")

    Q, H, center, feature = model(inputs)

    Q = Q.to("cpu").detach().numpy()
    labels = H.to("cpu").detach().numpy()
    feature = feature.to("cpu").detach().numpy()
    center = center.to("cpu").detach().numpy()

    return Q, labels, center, feature


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight",
        "-w",
        default=
        '../../ssn-logs/pointnetx-inst-spix-4-22/ep_97_batch_218_iter_59000_asa_0.942_ue_0.115.pth',
        type=str,
        help="/path/to/pretrained_weight")
    parser.add_argument("--fdim",
                        "-d",
                        default=20,
                        type=int,
                        help="embedding dimension")
    parser.add_argument("--niter",
                        "-n",
                        default=10,
                        type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix",
                        default=50,
                        type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", "-p", default=10, type=float)
    parser.add_argument("--folder",
                        "-f",
                        default='log',
                        help="a folder to store result")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    # data = shapenet_cpt("lib/shapenet_cpt_test0.h5")
    data = shapenet_man("../../data")
    loader = DataLoader(data, batch_size=1, shuffle=False)
    model = PointNet_SSNx_old(args.fdim,
                         args.nspix,
                         args.niter,
                         backend=soft_slic_pknn).to("cuda")
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    print(model)

    s = time.time()

    for i, (pointcloud, _, _) in enumerate(loader):
        print(i)
        pointcloud = pointcloud.to("cuda")
        _, labels, _, _ = inference(pointcloud, 10, model)
        labels = labels.transpose(1, 0)
        pointcloud = pointcloud.squeeze(0).detach().to("cpu").transpose(
            1, 0).numpy()
        #spix = spix.squeeze(0).transpose(1,  0).numpy()
        ptcloud = np.concatenate((pointcloud, labels, labels, labels), axis=-1)
        write.tobcd(ptcloud, 'xyzrgb',
                    os.path.join(args.folder, '{}.pcd'.format(i)))
