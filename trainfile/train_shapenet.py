import os
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime

from ..lib.utils.meter import Meter
from ..lib.ssn.ssn import soft_slic_pknn
from ..models.model_ptnet import PointNet_SSN
from ..lib.dataset import shapenet
from ..lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from ..lib.utils.pointcloud_io import CalAchievableSegAccSingle, CalUnderSegErrSingle


@torch.no_grad()
def eval(model, loader, pos_scale, device):
    model.eval()  # change the mode of model to eval
    sum_asa = 0
    sum_usa = 0
    cnt = 0
    for data in loader:
        cnt += 1
        inputs, _, labels_num = data  # b*c*npoint

        inputs = inputs.to(device)  # b*c*w*h
        # labels = labels.to(device)  # sematic_lable
        inputs = pos_scale * inputs
        # calculation,return affinity,hard lable,feature tensor
        _, H, _, _ = model(inputs)
        H = H.squeeze(0).to("cpu").detach().numpy()
        labels_num = labels_num.squeeze(0).numpy()
        asa = CalAchievableSegAccSingle(H, labels_num)
        usa = CalUnderSegErrSingle(H, labels_num)
        sum_asa += asa
        sum_usa += usa
        if (100 == cnt):
            break
    model.train()
    asaa = sum_asa / 100.0
    usaa = sum_usa / 100.0
    strs = "[test]:asa: {:.5f},ue: {:.5f}".format(asaa, usaa)
    print(strs)
    return asaa, usaa  # cal asa


def update_param(data, model, optimizer, compactness, pos_scale, device):
    inputs, labels, _ = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    inputs = pos_scale * inputs

    Q, H, _, _ = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, inputs, H)
    #uniform_compactness = uniform_compact_loss(Q,coords.reshape(*coords.shape[:2], -1), H,device=device)

    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {
        "loss": loss.item(),
        "reconstruction": recons_loss.item(),
        "compact": compact_loss.item(),
        "lr": optimizer.state_dict()['param_groups'][0]['lr']
    }


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = PointNet_SSN(cfg.fdim,
                         cfg.nspix,
                         cfg.niter,
                         backend=soft_slic_pknn).to(device)

    optimizer = optim.Adam(model.parameters(), cfg.lr)
    decayer = optim.lr_scheduler.StepLR(optimizer, 2, 0.94)

    train_dataset = shapenet.shapenet(cfg.root)
    train_loader = DataLoader(train_dataset,
                              cfg.batchsize,
                              shuffle=True,
                              drop_last=True,
                              num_workers=cfg.nworkers)

    test_dataset = shapenet.shapenet(cfg.root, split="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    iterations = 0
    writer = SummaryWriter(log_dir=args.out_dir, comment='traininglog')
    for epoch_idx in range(cfg.train_epoch):
        batch_iterations = 0
        for data in train_loader:
            batch_iterations += 1
            iterations += 1
            metric = update_param(data, model, optimizer, cfg.compactness,
                                  cfg.pos_scale, device)
            meter.add(metric)
            state = meter.state(
                f"[{batch_iterations},{epoch_idx}/{cfg.train_epoch}]")
            print(state)
            addscaler(metric, writer, iterations)
            if (iterations % 200) == 0:
                test_res = eval(model, test_loader, cfg.pos_scale, device)
                addscaler(test_res, writer, iterations, True)
                if (iterations % 1000) == 0:
                    (asa, usa) = test_res
                    strs = "ep_{:}_batch_{:}_iter_{:}_asa_{:.3f}_ue_{:.3f}.pth".format(
                        epoch_idx, batch_iterations, iterations, asa, usa)
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.out_dir, strs))
        decayer.step()
    unique_id = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, "model" + unique_id + ".pth"))

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, "model" + unique_id + ".pth"))


def addscaler(metric, scalarWriter, iterations, test=False):
    if not test:
        scalarWriter.add_scalar("comprehensive/loss", metric["loss"],
                                iterations)
        scalarWriter.add_scalar("loss/reconstruction_loss",
                                metric["reconstruction"], iterations)
        scalarWriter.add_scalar("loss/compact_loss", metric["compact"],
                                iterations)
        scalarWriter.add_scalar("lr", metric["lr"], iterations)
    else:
        (asa, usa) = metric
        scalarWriter.add_scalar("eval/asa", asa, iterations)
        scalarWriter.add_scalar("eval/ue", usa, iterations)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root",
                        type=str,
                        default='../shapenet_part_seg_hdf5_data',
                        help="/ path/to/shapenet")
    parser.add_argument("--out_dir",
                        default="../ssn-logs/pointnet-pknn-",
                        type=str,
                        help="/path/to/output directory")
    parser.add_argument("--batchsize", default=20, type=int)
    parser.add_argument("--nworkers",
                        default=8,
                        type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--train_epoch", default=100, type=int)
    parser.add_argument("--fdim",
                        default=20,
                        type=int,
                        help="embedding dimension")
    parser.add_argument("--niter",
                        default=10,
                        type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix",
                        default=50,
                        type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", default=10, type=float)
    parser.add_argument("--compactness", default=1e-4, type=float)
    parser.add_argument("--test_interval", default=100, type=int)

    args = parser.parse_args()
    date = datetime.now()
    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
