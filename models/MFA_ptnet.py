import torch
import torch.nn as nn
from lib.ssn.ssn import soft_slic_pknn
from lib.MEFEAM.MEFEAM import MultiScaleFeatureAggregation, mlp
from lib.utils.pointcloud_io import CalAchievableSegAccSingle, CalUnderSegErrSingle
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from ..lib.pointnet.pointnet import ptnet


class MfaPtnetSsn(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10, RGB=False, normal=False, backend=soft_slic_pknn):
        super().__init__()
        self.channel = 3
        if RGB:
            self.channel += 3
        if normal:
            self.channel += 3
        self.feature_dim = feature_dim
        self.backend = backend
        self.nspix = nspix
        self.n_iter = n_iter
        self.mfa = MultiScaleFeatureAggregation(
            [32, 64, 64], [128, 64, self.feature_dim], 32, self.channel, [0.2, 0.3, 0.4])
        self.ptnet = ptnet(self.feature_dim)
        self.fushion = mlp([2*feature_dim, feature_dim], feature_dim)

    def forward(self, x):
        mfa = self.mfa(x)
        ptnet = self.ptnet(x)
        net = torch.cat((ptnet, mfa), dim=1)
        net = self.fushion(net)
        return soft_slic_pknn(net, net[:, :, :self.nspix]), net


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
        (_, H, _, _), _ = model(inputs)
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
    inputs, labels, labels_num = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    inputs = pos_scale * inputs

    (Q, H, _, _), _ = model(inputs)

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
    }


def addscaler(metric, scalarWriter, iterations, test=False):
    if not test:
        scalarWriter.add_scalar("comprehensive/loss",
                                metric["loss"], iterations)
        scalarWriter.add_scalar("loss/reconstruction_loss",
                                metric["reconstruction"], iterations)
        scalarWriter.add_scalar("loss/compact_loss", metric["compact"],
                                iterations)
    else:
        (asa, usa) = metric
        scalarWriter.add_scalar("test/asa", asa, iterations)
        scalarWriter.add_scalar("test/ue", usa, iterations)
