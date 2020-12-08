import torch
import torch.nn as nn
from torch.nn.functional import relu
from utils.util_funcs import knn_indices_func_cpu, knn_indices_func_gpu

# square_distance,query_ball point and fps sampling comes from github repository Pointnet_Point2_Pytorch
# minor changes are made to fit [B,C,N] tensors instead of[B,N,C]
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.bmm(src.permute(0, 2, 1,), dst)
    dist += torch.sum(src ** 2, 1).view(B, N, 1)
    dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    return dist.squeeze(-1)


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, 3, N]
        new_xyz: query points, [B, 3, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(
        device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group_query_ball(radius, nsample, xyz, points, use_xyz):
    """sample and group points to  fit MFEM input using query ball sampling

    Input:
        radius (float): search radius if using query_ball
        nsample (int): max point query_ball and npoint in knn sampling
        xyz (Tensor,[B,npoint,3]): xyz corrdinate
        points (Tensor,[B,npoint,channel]): point to be grouped
        use_xyz (bool): if the function concat xyz with grouped points
    Output:
        new_points  (Tensor,[B,npoint,nsample,channel]) grouped output
    """
    idx = query_ball_point(radius, nsample, xyz, points)
    grouped_points = torch.stack(
        [torch.stack([x[:, idxxx] for idxxx in idxx], dim=1) for idxx, x in zip(idx, points)])

    if use_xyz:
        grouped_xyz = xyz.unsquezze(2).repeat(1, 1, nsample, 1)
        return torch.cat([grouped_xyz, grouped_points], -1)
    else:
        return grouped_points


def sample_and_group_knn(radius, nsample, xyz, points, use_xyz):
    """sample and group points to  fit MFEM input using knn

    Input:
        radius (float): deserve nothing,just to keep the same api as sample_and_group_query_ball
        nsample (int): number of point knn aggregated
        xyz (Tnsor,[B,S,channel]): input point center
        points (Tensor,[B,npoint,channel]): point to be grouped
        use_xyz (Bool): if the function concat xyz with grouped points
    """

    if point.device = 'cpu':
        knn = knn_indices_func_cpu
    else:
        knn = knn_indices_func_gpu

    idx = knn(xyz, points, nsample, 1)
    grouped_points = torch.stack(
        [torch.stack([x[:, idxxx] for idxxx in idxx], dim=1) for idxx, x in zip(idx, points)])

    if use_xyz:
        grouped_xyz = xyz.unsquezze(2).repeat(1, 1, nsample, 1)
        return torch.cat([grouped_xyz, grouped_points], -1)
    else:
        return grouped_points


class mlp(nn.Module):
    """
    standalone mlp module to construct MFEAM and LFAM

    Args:
        mlp (list): the structure of mlp
        channel_in(int): number of input channel
        kernel_start:tuple,to control first kernel in mlp
        name(str):name of the mlp
        activation (torch function): activation function suh as relu
        bn (bool): if the mlp uses bn
    """

    def __init__(self, mlp, channel_in, kernel_start=(1, 1), name='mlp', activation=None, bn=False):
        self.mlp_ = nn.Sequential()
        self.name = name
        self.pre_channel = channel_in
        for i, layer in enumerate(mlp):
            if i == 0:
                self.mlp_.add_module(self.name + '_' + str(i), nn.Conv2d(
                    self.pre_channel, layer, kernel_start, (1, 1), bias=True))
            else:
                self.mlp_.add_module(
                    self.name + '_' + str(i), nn.Conv2d(self.pre_channel, layer, (1, 1), (1, 1), bias=True))
            if activation is not None:
                self.mlp_.add_module(
                    self.name + "_activation_" + str(i), activation())
        if bn:
            self.mlp_.add_module(
                self.name + "_bn", nn.BatchNorm1d(self.pre_channel))

    def forward(self, x):
        return self.mlp_(x)


class MFEM(nn.Module):
    """
    MFEM module
    Args:
        mlp_multiscale (list): sequence to construct multi_scale_mlp
        mlp_global(list):sequence to construct mlp for global feature
        mlp_msf(list):sequence to construct mlp for Multi-Scale Locality Feature Spaces
        nsample (int): knn sample
        channel_in (int): input channel
        point_scale (list): scales
        grouping (function): query_ball or knn
        kernel_start (tuple, optional): first kernel in mlp,Defaults to (1, 1).
    """

    def __init__(self, mlp_multiscale, mlp_global, mlp_msf, nsample, channel_in, point_scale, grouping=sample_and_group_knn, kernel_start=(1, 1)):

        #sample_and_group_knn(radius, nsample, xyz, points, use_xyz)
        self.sample_and_group = grouping
        self.nsample = nsample

        self.scale0 = mlp(mlp_multiscale, channel_in,
                          kernel_start, 'scale0', nn.ReLU)
        self.scale1 = mlp(mlp_multiscale, channel_in,
                          kernel_start, 'scale1', nn.ReLU)
        self.scale2 = mlp(mlp_multiscale, channel_in,
                          kernel_start, 'scale2', nn.ReLU)

        self.mlp_global = mlp(
            mlp_global, mlp_multiscale[-1], name='global', activation=nn.ReLU)

        self.mlp_MSF = mlp(
            mlp_msf, mlp_global[-1], name="msf", activation=nn.ReLU)

    def forward(self, x):
        #[B,C,N]->[B,C,nsample,N]
        point0 = self.sample_and_group(
            point_scale[0], self.nsample, x, x, False)
        point1 = self.sample_and_group(
            point_scale[1], self.nsample, x, x, False)
        point2 = self.sample_and_group(
            point_scale[2], self.nsample, x, x, False)

        # ->[B,C,nsample,N]
        point0 = self.scale0(point0)
        point1 = self.scale1(point1)
        point2 = self.scale2(point2)

        # ->[B,C,N]
        point0 = point0.max(2)
        point1 = point1.max(2)
        point2 = point2.max(2)

        # ->[B,C]
        # \->[B,m,N]
        point = torch.cat([point0, point1, point2], dim=-1)
        point = self.mlp_global(point)
        global_feature = torch.max(global_feature, dim=2)
        msf_feature = self.mlp_MSF(global_feature)
        return global_feature, msf_feature


class LFAM(nn.module):
    """
    LFAM module in MFEAM
    Args:
        nsample (int): nsample in sample and group
        grouping (function):grouping method,knn or query_ball
        mlp (list): mlp structure used for feature fusion
        channel_in (int): input feature channel
    """
    def __init__(self, nsample, grouping, mlp, channel_in,):
        self.sample_and_group = grouping
        self.nsample = nsample
        self.mlp = mlp(mlp, channel_in, name='output mlp', activation=nn.ReLU)

    def forward(self, global_feature, msf):
        msf_grouped = self.sample_and_group(1, self.nsample, msf, msf)
        contacted_feature = torch.cat(
            [global_feature.repeat(1, 1, self.nsample, msf.shape[-1], msf], dim=1))
        mixed_feature = self.mlp(contacted_feature)
        maxed_feature = mixed_feature.max(2)

        return maxed_feature


class discriminative_loss(nn.Module):
    """
    reimplemention of discriminative loss in ASIS using Pytorch\n
    ASIS paper here:https://arxiv.org/abs/1902.09852 \n
    Origional implemention here(using tensorflow):https://github.com/WXinlong/ASIS.git

    Args:
        d_var (float): theta_V in L_var
        d_dist ([type]): theta_d in L_dist
        param_var (float, optional): param before var loss. Defaults to 1.0.
        param_dist (float, optional): param before dist loss. Defaults to 1.0.
        param_reg (float, optional): param before regression loss. Defaults to 0.001.
    """

    def __init__(self, d_var, d_dist, param_var=1.0, param_dist=1.0, param_reg=0.001):
        super().__init__()
        self.var = var_loss(d_var)
        self.dist = dist_loss(d_diff)
        self.par_var = param_var
        self.par_dist = param_dist
        self.par_reg = param_reg

        def forward(self, prediction, label):
        l_reg = 0
        for batch in range(prediction.shape[0]):
            center, unique = cal_center(prediction, label, batch)
            l_reg += torch.sum(torch.norm(center, dim=0)) / unique

        return self.par_var*self.var(prediction, label) + self.par_dist*self.dist(prediction, label)+self.par_reg*l_reg

        def cal_center(prediction, label, batch):
        unique = torch.unique(label[batch])
        return torch.stack([torch.mean(prediction[batch, :, label[batch] == unique_label], dim=-1, keepdim=True)for unique_label in unique]), unique.shape[0]

    class var_loss(nn.Module):
        def __init__(self, d_var):
            super().__init__()
            self.d_var = d_var

        def forward(prediction, label):
            ret = 0
            for batch in range(prediction.shape[0]):
                unique = torch.unique(label[batch])
                instance_loss = 0
                for unique_label in unique:
                    mask = label[batch] == unique_label
                    point = prediction[batch, :, mask]
                    center = torch.mean(point, dim=0)
                    instance_loss += torch.sum(relu(torch.norm(point -
                                                               center, dim=0)-self.d_var)**2)/point.shape[-1]
                ret += instance_loss/unique.shape[0]

            return ret

    class dist_loss(nn.Module):
        def __init__(self, d_dist):
            self.d_dist = d_dist

        def forward(prediction, label):
            ret = 0
            for batch in range(data.shape[0]):
                loss = 0
                center, unique = cal_center(prediction, label)
                print(center.shape)
                square_matrix = square_distance_single(center, center)
                dist_matrix = torch.sqrt(square_matrix)
                loss += torch.sum(relu(2*self.d_dist-dist_matrix)**2) /
                (2 * unique * (unique - 1))
            return loss

        def square_distance_single(src, dst):
        """
        Calculate Euclid distance between each two points.
        single batch version,to cal the dist between two points

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [C, N]
            dst: target points, [C, M]
        Output:
            dist: per-point square distance, [N, M]
        """
        _, N = src.shape
        _, M = dst.shape
        dist = -2 * torch.matmul(src.permute(1, 0), dst)
        dist += torch.sum(src ** 2, 0).view(N, 1)
        dist += torch.sum(dst ** 2, 0).view(1, M)
        return dist.squeeze(-1)
