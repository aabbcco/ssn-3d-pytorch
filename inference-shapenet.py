import math
import numpy as np
import torch
import os

from skimage.color import rgb2lab
from lib.dataset.shapenet import shapenet, shapenet_spix
from lib.utils.pointcloud_io import write
from torch.utils.data import DataLoader
from lib.ssn.ssn import soft_slic_all
from model_ptnet import PointNet_SSN,PointNet_SSKNN


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
    parser.add_argument("--weight","-w", default='log/model-shapenet.pth',
                        type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim","-d", default=10, type=int,
                        help="embedding dimension")
    parser.add_argument("--niter","-n", default=10, type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=50, type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale","-p", default=10, type=float)
    parser.add_argument("--folder","-f",default='log',help="a folder to store result")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    data = shapenet("../shapenet_part_seg_hdf5_data",
                    split='val')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    model = PointNet_SSKNN(args.fdim, args.nspix, args.niter).to("cuda")
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    print(model)

    s = time.time()

    for i, (pointcloud, label,labell) in enumerate(loader):
        print(i)
        _, labels, _, _ = inference(pointcloud,10,model)

        pointcloud = pointcloud.squeeze(0).transpose(1, 0).numpy()
        label = labell.transpose(1, 0).numpy()
        #spix = spix.squeeze(0).transpose(1,  0).numpy()
        ptcloud = np.concatenate(
            (pointcloud, label, label, labels.transpose(1, 0)),  axis=-1)
        write.tobcd(ptcloud,  'xyzrgb', os.path.join(args.folder,'{}.pcd'.format(i)))
        # Q, label, center, feature = inference(pointcloud, args.nspix, args.niter,
        #                           args.fdim, args.pos_scale, args.weight)
        # print(f"time {time.time() - s}sec")
        # np.savetxt("Q.csv", np.squeeze(Q, 0), fmt="%.8e", delimiter=",")
        # np.savetxt("center.csv", np.squeeze(center, 0), fmt="%.10e", delimiter=",")
        # np.savetxt("feature.csv" np.squeeze(feature, 0), fmt="%.10e", delimiter=",")
        # ptcloud = pointcloud.squeeze(0).permute(1, 0).numpy()
        # ptcloud = np.concatenate((ptcloud, label.transpose(1, 0)), axis=-1)
        # write.tobcd(ptcloud, 'xyzl', 'shapenet_pred.pcd')
