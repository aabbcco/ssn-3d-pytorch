import os
import math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.bistream import bistream_SSN
from lib.utils.meter import Meter
from lib.ssn.ssn import soft_slic_pknn
from lib.dataset import shapenet
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from lib.MEFEAM.MEFEAM import discriminative_loss
from lib.utils.pointcloud_io import CalAchievableSegAccSingle, CalUnderSegErrSingle


@torch.no_grad()
def eval(model, loader, pos_scale, device):
    model.eval()  # change the mode of model to eval
    sum_asa = 0
    sum_usa = 0
    cnt = 0
    for data in loader:
        cnt += 1
        inputs, labels, labels_num = data  # b*c*npoint

        inputs = inputs.to(device)  # b*c*w*h
        #labels = labels.to(device)  # sematic_lable
        inputs = pos_scale * inputs
        # calculation,return affinity,hard lable,feature tensor
        (Q, H, _, _), msf_feature = model(inputs)
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


def update_param(data, model, optimizer, compactness, pos_scale, device,
                 disc_loss):
    inputs, labels, labels_num = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    inputs = pos_scale * inputs

    (Q, H, _, _), msf = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, inputs, H)
    disc = disc_loss(msf, labels_num)
    #uniform_compactness = uniform_compact_loss(Q,coords.reshape(*coords.shape[:2], -1), H,device=device)

<<<<<<< HEAD
    loss = recons_loss + compactness * compact_loss + 0.001 * disc
=======
    loss = recons_loss + compactness * compact_loss+1e-3*disc
>>>>>>> ad59c2986a8214b758c7150396c185013fc7837b

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {
        "loss": loss.item(),
        "reconstruction": recons_loss.item(),
        "disc_loss":(1e-3 *disc).item(),
        "compact": compact_loss.item(),
    }


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = bistream_SSN(cfg.fdim,
                         cfg.nspix,
                         cfg.niter,
                         backend=soft_slic_pknn).to(device)

    disc_loss = discriminative_loss(0.1, 0.5,1e-4)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

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
    writer = SummaryWriter(log_dir=cfg.out_dir, comment='traininglog')
    for epoch_idx in range(cfg.train_epoch):
        batch_iterations = 0
        for data in train_loader:
            batch_iterations += 1
            iterations += 1
            metric = update_param(data, model, optimizer, cfg.compactness,
                                  cfg.pos_scale, device, disc_loss)
            meter.add(metric)
            state = meter.state(
                f"[{batch_iterations},{epoch_idx}/{cfg.train_epoch}]")
            print(state)
            # return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}
            writer.add_scalar("comprehensive/loss", metric["loss"], iterations)
            writer.add_scalar("loss/reconstruction_loss",
                              metric["reconstruction"], iterations)
            writer.add_scalar("loss/compact_loss", metric["compact"],
                              iterations)
            writer.add_scalar("loss/disc_loss", metric["disc_loss"],
                              iterations)
            if (iterations % 200) == 0:
                (asa, usa) = eval(model, test_loader, cfg.pos_scale, device)
                writer.add_scalar("test/asa", asa, iterations)
                writer.add_scalar("test/ue", usa, iterations)
                if (iterations % 1000) == 0:
                    strs = "ep_{:}_batch_{:}_iter_{:}_asa_{:.3f}_ue_{:.3f}.pth".format(
                        epoch_idx, batch_iterations, iterations, asa, usa)
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            cfg.out_dir, strs))
            # if (iterations % cfg.test_interval) == 0:
            #     asa = eval(model, test_loader, cfg.pos_scale,  device)
            #     print(f"validation asa {asa}")
            #     writer.add_scalar("comprehensive/asa", asa, iterations)
            #     if asa > max_val_asa:
            #         max_val_asa = asa
            #         torch.save(model.state_dict(), os.path.join(
            #             cfg.out_dir, "bset_model_sp_loss.pth"))

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, "model" + unique_id + ".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root",
                        type=str,
                        default='../shapenet_part_seg_hdf5_data',
                        help="/ path/to/shapenet")
    parser.add_argument("--out_dir",
                        default="./log_bistream_ndisc",
                        type=str,
                        help="/path/to/output directory")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--nworkers",
                        default=8,
                        type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
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
                        default=100,
                        type=int,
                        help="number of superpixels")
    parser.add_argument("--pos_scale", default=10, type=float)
    parser.add_argument("--compactness", default=1e-4, type=float)
    parser.add_argument("--test_interval", default=100, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
