import torch
import torch.nn as nn
from lib.pointnet.pointnet import STN3d, STNkd
import torch.nn.functional as F
from lib.ssn.ssn import soft_slic_pknn
from lib.MEFEAM.MEFEAM import MFEM, mlp, sample_and_group_query_ball
from lib.utils.pointcloud_io import CalAchievableSegAccSingle, CalUnderSegErrSingle
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse


class ptnet(nn.Module):
    def __init__(
        self,
        feature_dim,
        RGB=False,
        normal=False,
    ):
        """
        Spix Network Using pointnet as frontend

        Args:
            feature_dim (int): dim of output feature
            nspix (int): getwork generate n spixs
            n_iter (int, optional): n soft silc iters. Defaults to 10.
            RGB (bool, optional): if the rgb feature is used. Defaults to False.
            normal (bool, optional): if the normal feature is used. Defaults to False.
            backend (function, optional): a backend soft slic function. Defaults to soft_slic_knn.
        """
        super().__init__()
        self.channel = 3
        if RGB:
            self.channel += 3

        if normal:
            self.channel += 3

        self.stn = STN3d(self.channel)
        self.conv1 = torch.nn.Conv1d(self.channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4928, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, feature_dim, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net


class multi_ptnet_SSN(nn.Module):
    def __init__(self,
                 feature_dim,
                 nspix,
                 mfem_dim=6,
                 n_iter=10,
                 RGB=False,
                 Normal=False,
                 backend=soft_slic_pknn):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.feature_dim = feature_dim
        self.backend = backend
        self.mfem = MFEM([32, 64], [128, 128], [64, mfem_dim], 32, 3,
                         [0.2, 0.3, 0.4], sample_and_group_query_ball)
        self.ptnet = ptnet(self.feature_dim)
        self.mpl_fusion = mlp([self.feature_dim * 2, self.feature_dim],
                              self.feature_dim * 2,
                              activation=nn.ReLU,
                              bn=True)

    def forward(self, x):
        ptnet_feature = self.ptnet(x)
        global_feature, msf_feature = self.mfeam(x)
        combined_feature = torch.cat([ptnet_feature, msf_feature], dim=1)
        fusioned_feature = self.mpl_fusion(combined_feature)
        return self.backend(fusioned_feature,
                            fusioned_feature[:, :, :self.nspix],
                            self.n_iter), msf_feature


# training

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
        # labels = labels.to(device)  # sematic_lable
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

    loss = recons_loss + compactness * compact_loss + 0.05 * disc

    optimizer.zero_grad()  # clear previous grad
    loss.backward()  # cal the grad
    optimizer.step()  # backprop

    return {
        "loss": loss.item(),
        "reconstruction": recons_loss.item(),
        "disc_loss": disc,
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
        scalarWriter.add_scalar("loss/disc_loss", metric["disc_loss"],
                                iterations)
    else:
        (asa, usa) = metric
        scalarWriter.add_scalar("test/asa", asa, iterations)
        scalarWriter.add_scalar("test/ue", usa, iterations)
