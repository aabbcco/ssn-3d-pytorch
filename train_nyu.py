import os
import math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.utils.meter import Meter
from model import SSNModel
from lib.dataset import NYUv2, augmentation
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse, uniform_compact_loss


@torch.no_grad()
def eval(model, loader, color_scale, pos_scale, device, num_sample=50):
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
    for i in range(num_sample):
        inputs, labels, _, _ = iter(loader).next()  # b*c*w*h

        inputs = inputs.to(device)  # b*c*w*h
        labels = labels.to(device)  # sematic_lable

        height, width = inputs.shape[-2:]  # w*H

        # determine how much pix initiated in horizontal/vertical space
        nspix_per_axis = int(math.sqrt(model.nspix))
        pos_scale = pos_scale * \
            max(nspix_per_axis / height, nspix_per_axis / width)  # dont konw

        # add coords for each pixel,B*(C+2)*W*H
        coords = torch.stack(torch.meshgrid(torch.arange(
            height, device=device), torch.arange(width, device=device)), 0)  # 2*W*H?
        # whats the meaning of repeat?#B*2*W*H
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        # input data, B*(C+2(x,y))*W*H
        inputs = torch.cat([color_scale*inputs, pos_scale*coords], 1)

        # calculation,return affinity,hard lable,feature tensor
        Q, H, feat = model(inputs)

        H = H.reshape(height, width)  # B*(W*H)=>B*W*H
        labels = labels.argmax(1).reshape(height, width)  # one hot to digit?

        asa = achievable_segmentation_accuracy(H.to("cpu").detach(
        ).numpy(), labels.to("cpu").numpy())  # return data to cpu
        sum_asa += asa
    model.train()
    return sum_asa / len(loader)  # cal asa


def update_param(data, model, optimizer, spix_weight, compactness, color_scale, pos_scale, device):
    inputs, labels, spix, _ = data

    inputs = inputs.to(device)
    labels = labels.to(device)
    spix = spix.to(device)

    height, width = inputs.shape[-2:]

    # determine the origion poisition of superpixel
    nspix_per_axis = int(math.sqrt(model.nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    coords = torch.stack(torch.meshgrid(torch.arange(
        height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([color_scale*inputs, pos_scale*coords], 1)

    Q, H, _ = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    recons_loss_spix = reconstruct_loss_with_cross_etnropy(Q, spix)
    compact_loss = reconstruct_loss_with_mse(
        Q, coords.reshape(*coords.shape[:2], -1), H)
    #uniform_compactness = uniform_compact_loss(Q,coords.reshape(*coords.shape[:2], -1), H,device=device)

    loss = recons_loss + spix_weight*recons_loss_spix + compactness * compact_loss

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "spix": recons_loss_spix.item(), "compact": compact_loss.item()}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = SSNModel(cfg.fdim, cfg.nspix, cfg.niter).to(device)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(
    ), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = NYUv2.NYUv2(cfg.root, geo_transforms=augment)
    train_loader = DataLoader(train_dataset, cfg.batchsize,
                              shuffle=True, drop_last=True, num_workers=cfg.nworkers)

    test_dataset = NYUv2.NYUv2(cfg.root, split="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=True, drop_last=False)

    meter = Meter()

    iterations = 0
    max_val_asa = 0
    writer = SummaryWriter(log_dir='log', comment='traininglog')
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(
                data, model, optimizer, cfg.spix_weight, cfg.compactness, cfg.color_scale, cfg.pos_scale,  device)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            # return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item(),"uniform:":uniform_compactness.item()}
            writer.add_scalar("comprehensive/loss", metric["loss"], iterations)
            writer.add_scalar("loss/reconstruction_loss",
                              metric["reconstruction"], iterations)
            writer.add_scalar("loss/compact_loss",
                              metric["compact"], iterations)
            writer.add_scalar("loss/spix_loss", metric["spix"], iterations)
            if (iterations % cfg.test_interval) == 0:
                asa = eval(model, test_loader, cfg.color_scale,
                           cfg.pos_scale,  device)
                print(f"validation asa {asa}")
                writer.add_scalar("comprehensive/asa", asa, iterations)
                if asa > max_val_asa:
                    max_val_asa = asa
                    torch.save(model.state_dict(), os.path.join(
                        cfg.out_dir, "bset_model_sp_loss.pth"))
            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(), os.path.join(
        cfg.out_dir, "model"+unique_id+".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default='../NYUv2', help="/ path/to/NYUv2")
    parser.add_argument("--out_dir", default="./log",
                        type=str, help="/path/to/output directory")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--nworkers", default=4, type=int,
                        help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=200000, type=int)
    parser.add_argument("--fdim", default=20, type=int,
                        help="embedding dimension")
    parser.add_argument("--niter", default=5, type=int,
                        help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int,
                        help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--compactness", default=1e-4, type=float)
    parser.add_argument("--test_interval", default=1000, type=int)
    parser.add_argument('--spix_weight', default=0.5, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
