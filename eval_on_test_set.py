import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from skimage.color.colorconv import lab2rgb

from matplotlib import pyplot as plt

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter
from lib.dataset import bsds
from evaluations import undersegmentation_error, achievable_segmentation_accuracy, compactness, boundary_recall


@torch.no_grad()
def inference(image, nspix, n_iter, model, fdim=None, color_scale=0.26, pos_scale=2.5, enforce_connectivity=True):
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
    # if weight is not None:
    #     from model import SSNModel
    #     model = SSNModel(fdim, nspix, n_iter).to("cuda")
    #     model.load_state_dict(torch.load(weight))
    #     model.eval()
    # else:
    #     model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[2:]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    coords = torch.stack(torch.meshgrid(torch.arange(
        height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = image.to("cuda").float()

    inputs = torch.cat([color_scale*image, pos_scale*coords], 1)

    _, H, _ = model(inputs)

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default='../BSR', help="/path/to/val")
    parser.add_argument("--weight", default=None, type=str,
                        help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int,
                        help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int,
                        help="number of superpixels,100 to nspix")
    parser.add_argument('--dest', '-d', default='results',
                        help='dest folder the image saves')
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()

    weight = args.weight
    nspix = args.nspix
    n_iter = args.niter
    fdim = args.fdim

    # Dataset did everything for us
    dataset = bsds.BSDS(args.root, split='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # generate number of spix from 100 to nspix asstep 100
    for i in range(100, args.nspix+1, 100):
        if not os.path.exists(args.dest):
            os.mkdir(args.dest)

        if not os.path.exists(os.path.join(args.dest, str(i))):
            os.mkdir(os.path.join(args.dest, str(i)))

        if weight is not None:
            from model import SSNModel
            model = SSNModel(fdim, i, n_iter).to("cuda")
            model.load_state_dict(torch.load(weight))
            model.eval()
        else:
            def model(data): return sparse_ssn_iter(data, i, n_iter)

        # throw every image into the net
        for data in dataloader:
            image, label, name = data
            height, width = image.shape[-2:]
            label_pred = inference(image, args.nspix, args.niter,
                                   model, args.fdim, args.color_scale, args.pos_scale)
            label = label.argmax(1).reshape(height, width).numpy()
            np.savetxt(os.path.join(args.dest,
                                    str(i), name[0]+'.csv'), label_pred, fmt='%d', delimiter=',')
            asa = achievable_segmentation_accuracy(label_pred, label)
            usa = undersegmentation_error(label_pred, label)
            cptness = compactness(label_pred)
            BR = boundary_recall(label_pred, label)
            image = np.squeeze(image.numpy(), axis=0).transpose(1, 2, 0)
            image = lab2rgb(image)
            print(name[0], '\tprocessed,asa_{:.4f}_usa{:.4f}_co{:.4f}_BR_{:.4f}'.format(
                asa, usa, cptness, BR))
            plt.imsave(os.path.join(args.dest, str(i), "asa_{:.4f}_usa_{:.4f}_co_{:.4f}_BR_{:.4f}_{}.jpg".format(
                asa, usa, cptness, BR, name[0])), mark_boundaries(image, label_pred))
