import os
import sys
sys.path.append(os.path.dirname("../"))
import torch
import torch.nn as nn
from torch.nn.modules import module

from lib.pointnet.pointnet import STN3d, STNkd, feature_transform_reguliarzer
from lib.ssn.ssn import soft_slic_all, soft_slic_knn,soft_slic_pknn
import torch.nn.functional as F


class PointNet_SSN(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10, RGB=False, normal=False, backend=soft_slic_knn):
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
        self.nspix = nspix
        self.n_iter = n_iter
        self.channel = 3
        self.backend = backend
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
        concat = torch.cat([expand, out1, out2, out3, out4], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        #net = net.transpose(2, 1).contiguous()
        return self.backend(net, net[:, :, :self.nspix], self.n_iter)

class PointNet_SSNx(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10, RGB=False, normal=False, backend=soft_slic_knn):
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
        self.nspix = nspix
        self.n_iter = n_iter
        self.channel = 3
        self.backend = backend
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
        self.convs1 = torch.nn.Conv1d(2880, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128+self.channel, feature_dim, 1)
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
        concat = torch.cat([expand, out1, out2, out3, out4], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = torch.cat((point_cloud,net),dim=1)
        net = self.convs4(net)
        #net = net.transpose(2, 1).contiguous()
        return self.backend(net, net[:, :, :self.nspix], self.n_iter, k_point =192)


class PointNet_SSNx_old(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10, RGB=False, normal=False, backend=soft_slic_knn):
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
        self.nspix = nspix
        self.n_iter = n_iter
        self.channel = 3
        self.backend = backend
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
        self.convs4 = torch.nn.Conv1d(128+self.channel, feature_dim, 1)
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
        concat = torch.cat([expand, out1, out2, out3, out4,out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = torch.cat((point_cloud,net),dim=1)
        net = self.convs4(net)
        #net = net.transpose(2, 1).contiguous()
        return self.backend(net, net[:, :, :self.nspix], self.n_iter, k_point =192)

if __name__ == "__main__":
    model = PointNet_SSNx(20,soft_slic_pknn)
    print(model)
