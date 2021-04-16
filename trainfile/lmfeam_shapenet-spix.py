import os
import math
import numpy as np
import time
import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ..lib.utils.meter import Meter
from ..lib.dataset import shapenet, augmentation
from ..lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse, uniform_compact_loss
from ..lib.MEFEAM.MEFEAM import discriminative_loss, LMFEAM

from ..lib.ssn.ssn import soft_slic_pknn, soft_slic_all


class LMFEAM_SSN(Module):
    def __init__(self,
                 feature_dim,
                 nspix,
                 mfem_dim=6,
                 n_iter=10,
                 RGB=False,
                 normal=False,
                 backend=soft_slic_all):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.channel = 3
        self.backend = backend
        if RGB:
            self.channel += 3
        if normal:
            self.channel += 3
        #[32, 64], [128, 128], [64, mfem_dim], 32,3 , [0.2, 0.4, 0.6]
        self.lmfeam = LMFEAM([32, 64], [64, 128], [64, mfem_dim],
                             [128, 64, feature_dim],
                             32,
                             self.channel,
                             point_scale=[0.2, 0.4, 0.6])

    def forward(self, x):
        feature, msf = self.lmfeam(x)
        return self.backend(feature, feature[:, :, :self.nspix],
                            self.n_iter), msf


@torch.no_grad()
def eval(model, loader, pos_scale, device):
    def achievable_segmentation_accuracy(superpixel, label):
        """
        Function to calculate Achievable Segmentation Accuracy:
            ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

        Args:
            input: superpixel image (H, W),
            output: ground-truth (H, W)
        """
        TP = 0
        unique_id = np.unique(superpixel)
        for uid in unique_id:
            mask = superpixel == uid
            label_hist = np.histogram(label[mask])
            maximum_regionsize = label_hist[0].max()
            TP += maximum_regionsize
        return TP / label.size

    model.eval()  # change the mode of model to eval
    sum_asa = 0
    for data in loader:
        inputs, labels = data  # b*c*npoint

        inputs = inputs.to(device)  # b*c*w*h
        labels = labels.to(device)  # sematic_lable

        inputs = pos_scale * inputs
        # calculation,return affinity,hard lable,feature tensor
        Q, H, feat = model(inputs)

        asa = achievable_segmentation_accuracy(
            H.to("cpu").detach().numpy(),
            labels.to("cpu").numpy())  # return data to cpu
        sum_asa += asa
    model.train()
    return sum_asa / len(loader)  # cal asa


def update_param(data, model, optimizer, compactness, pos_scale, device,
                 disc_loss):
    inputs, labels, spix, spix_num = data

    inputs = inputs.to(device)
    labels = labels.to(device)
    spix = spix.to(device)

    inputs = pos_scale * inputs

    (Q, H, _, _), msf_feature = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    spix_loss = reconstruct_loss_with_cross_etnropy(Q, spix)
    compact_loss = reconstruct_loss_with_mse(Q, inputs, H)
    disc = disc_loss(msf_feature, spix_num)

    #uniform_compactness = uniform_compact_loss(Q,coords.reshape(*coords.shape[:2], -1), H,device=device)

    loss = recons_loss + 0.5 * spix_loss + compactness * compact_loss + disc

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {
        "loss": loss.item(),
        "spix": spix_loss.item(),
        "reconstruction": recons_loss.item(),
        "compact": compact_loss.item(),
        "disc": disc.item()
    }


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = LMFEAM_SSN(10, 50, backend=soft_slic_pknn).to(
        device)  # LMFEAM(10, 50).to(device)

    disc_loss = discriminative_loss(0.1, 0.1)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    train_dataset = shapenet.shapenet_spix(cfg.root)
    train_loader = DataLoader(train_dataset,
                              cfg.batchsize,
                              shuffle=True,
                              drop_last=True,
                              num_workers=cfg.nworkers)
    print(train_dataset.__len__())

    # test_dataset = shapenet.shapenet(cfg.root, split="test")
    # test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    iterations = 0
    writer = SummaryWriter(log_dir=cfg.out_dir, comment='traininglog')
    for epoch_idx in range(cfg.train_epoch):
        batch_iterations = 0
        for data in train_loader:
            iterations += 1
            batch_iterations += 1
            metric = update_param(data, model, optimizer, cfg.compactness,
                                  cfg.pos_scale, device, disc_loss)
            meter.add(metric)
            state = meter.state(
                f"[{batch_iterations},{epoch_idx}/{cfg.train_epoch}]")
            print(state)
            # return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}
            writer.add_scalar("comprehensive/loss", metric["loss"], iterations)
            writer.add_scalar("loss/spix_loss", metric["spix"], iterations)
            writer.add_scalar("loss/reconstruction_loss",
                              metric["reconstruction"], iterations)
            writer.add_scalar("loss/compact_loss", metric["compact"],
                              iterations)
            writer.add_scalar("loss/disc_loss", metric["disc"], iterations)
            if (iterations % 500) == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.out_dir, "model_epoch_" + str(epoch_idx) + "_" +
                        str(batch_iterations) + '_iter_' + str(iterations) +
                        ".pth"))

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, "model" + unique_id + ".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root",
                        type=str,
                        default='../shapenet_partseg_spix',
                        help="/ path/to/shapenet")
    parser.add_argument("--out_dir",
                        default="./log_lmnfeam_pknn-spix",
                        type=str,
                        help="/path/to/output directory")
    parser.add_argument("--batchsize", default=16, type=int)
    parser.add_argument("--nworkers",
                        default=8,
                        type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-6, type=float, help="learning rate")
    parser.add_argument("--train_epoch", default=30, type=int)
    parser.add_argument("--fdim",
                        default=10,
                        type=int,
                        help="embedding dimension")
    parser.add_argument("--niter",
                        default=5,
                        type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix",
                        default=100,
                        type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", default=10, type=float)
    parser.add_argument("--compactness", default=1e-4, type=float)
    parser.add_argument("--test_interval", default=100, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
