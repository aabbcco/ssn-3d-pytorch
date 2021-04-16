import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ..lib.utils.meter import Meter
from ..lib.ssn.ssn import soft_slic_pknn
from ..lib.dataset import shapenet
from ..lib.MEFEAM.MEFEAM import discriminative_loss
from ..models.MFA_ptnet import *


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = MfaPtnetSsn(cfg.fdim, cfg.nspix, cfg.niter,
                        backend=soft_slic_pknn).to(device)

    disc_loss = discriminative_loss(0.1, 0.5)

    optimizer = optim.Adam(model.parameters(), cfg.lr)
    dacayer = optim.lr_scheduler.StepLR(optimizer, 2, 0.9)

    train_dataset = shapenet.shapenet_inst(cfg.root)
    train_loader = DataLoader(train_dataset,
                              cfg.batchsize,
                              shuffle=True,
                              drop_last=True,
                              num_workers=cfg.nworkers)
.
    test_dataset = shapenet.shapenet_inst(cfg.root, split="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    iterations = 0
    writer = SummaryWriter(log_dir=cfg.out_dir, comment='traininglog')
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
                    torch.save(model.state_dict(),
                               os.path.join(cfg.out_dir, strs))
        dacayer.step()
    unique_id = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, "model" + unique_id + ".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root",
                        type=str,
                        default='../shapenet_partseg_inst',
                        help="/ path/to/shapenet")
    parser.add_argument("--out_dir",
                        default="./log_mfa_inst",
                        type=str,
                        help="/path/to/output directory")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--nworkers",
                        default=8,
                        type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_epoch", default=30, type=int)
    parser.add_argument("--fdim",
                        default=10,
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

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
