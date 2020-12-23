import os
import math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.utils.meter import Meter
from model_MNFEAM import MFEAM_SSN
from lib.dataset.shapenet import shapenet_spix
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse, uniform_compact_loss
from lib.MEFEAM.MEFEAM import discriminative_loss


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

        inputs = pos_scale*inputs
        # calculation,return affinity,hard lable,feature tensor
        Q, H, feat = model(inputs)

        asa = achievable_segmentation_accuracy(H.to("cpu").detach(
        ).numpy(), labels.to("cpu").numpy())  # return data to cpu
        sum_asa += asa
    model.train()
    return sum_asa / len(loader)  # cal asa


def update_param(data, model, optimizer, compactness,  pos_scale, device, disc_loss):
    inputs, labels, _, spix = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    inputs = pos_scale*inputs

    (Q, H, _, _), msf_feature = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(
        Q, inputs, H)
    disc = disc_loss(msf_feature, spix)

    #uniform_compactness = uniform_compact_loss(Q,coords.reshape(*coords.shape[:2], -1), H,device=device)

    loss = recons_loss + compactness * compact_loss + disc

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item(), "disc": disc.item()}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = MFEAM_SSN(10, 50).to(device)

    disc_loss = discriminative_loss(0.1, 0.5)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    train_dataset = shapenet_spix(cfg.root)
    train_loader = DataLoader(train_dataset, cfg.batchsize,
                              shuffle=True, drop_last=True, num_workers=cfg.nworkers)

    # test_dataset = shapenet.shapenet(cfg.root, split="test")
    # test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    iterations = 0
    max_val_asa = 0
    writer = SummaryWriter(log_dir='log', comment='traininglog')
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(
                data, model, optimizer, cfg.compactness, cfg.pos_scale,  device, disc_loss)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            # return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}
            writer.add_scalar("comprehensive/loss", metric["loss"], iterations)
            writer.add_scalar("loss/reconstruction_loss",
                              metric["reconstruction"], iterations)
            writer.add_scalar("loss/compact_loss",
                              metric["compact"], iterations)
            writer.add_scalar("loss/disc_loss", metric["disc"], iterations)
            if (iterations % 1000) == 0:
                torch.save(model.state_dict(), os.path.join(
                    cfg.out_dir, "model_iter"+str(iterations)+".pth"))
            # if (iterations % cfg.test_interval) == 0:
            #     asa = eval(model, test_loader, cfg.pos_scale,  device)
            #     print(f"validation asa {asa}")
            #     writer.add_scalar("comprehensive/asa", asa, iterations)
            #     if asa > max_val_asa:
            #         max_val_asa = asa
            #         torch.save(model.state_dict(), os.path.join(
            #             cfg.out_dir, "bset_model_sp_loss.pth"))
            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(), os.path.join(
        cfg.out_dir, "model"+unique_id+".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default= '../shapenet_partseg_spix' , help="/ path/to/shapenet")
    parser.add_argument("--out_dir", default="./log",
                        type=str, help="/path/to/output directory")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--nworkers", default=8, type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-6, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=10000, type=int)
    parser.add_argument("--fdim", default=10, type=int,
                        help="embedding dimension")
    parser.add_argument("--niter", default=5, type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", default=10, type=float)
    parser.add_argument("--compactness", default=1e-4, type=float)
    parser.add_argument("--test_interval", default=100, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)

    train(args)
