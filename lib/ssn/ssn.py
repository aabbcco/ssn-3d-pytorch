import math
import torch
from torch import softmax
from torch.nn.functional import pairwise_distance
from .util_funcs import knn_indices_func_cpu, knn_indices_func_gpu


#for time count->>SSN python toooooooooo slow
from time import time
SoftSlicTime = {}


# from .pair_wise_distance import PairwiseDistFunction
# from ..utils.sparse_utils import naive_sparse_bmm


# @torch.no_grad()
# def get_abs_indices(init_label_map, num_spixels_width):
#     b, n_pixel = init_label_map.shape
#     device = init_label_map.device
#     r = torch.arange(-1, 2.0, device=device)
#     relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

#     abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
#     abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
#     abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

#     return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


# @torch.no_grad()
# def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
#     relative_label = affinity_matrix.max(1)[1]
#     r = torch.arange(-1, 2.0, device=affinity_matrix.device)
#     relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
#     label = init_label_map + relative_spix_indices[relative_label]
#     return label.long()


# @torch.no_grad()
# def sparse_ssn_iter(pixel_features, num_spixels, n_iter):
#     """
#     computing assignment iterations with sparse matrix
#     detailed process is in Algorithm 1, line 2 - 6
#     NOTE: this function does NOT guarantee the backward computation.

#     Args:
#         pixel_features: torch.Tensor
#             A Tensor of shape (B, C, H, W)
#         num_spixels: int
#             A number of superpixels
#         n_iter: int
#             A number of iterations
#         return_hard_label: bool
#             return hard assignment or not
#     """
#     height, width = pixel_features.shape[-2:]
#     num_spixels_width = int(math.sqrt(num_spixels * width / height))
#     num_spixels_height = int(math.sqrt(num_spixels * height / width))

#     spixel_features, init_label_map = \
#         calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
#     abs_indices = get_abs_indices(init_label_map, num_spixels_width)

#     pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
#     permuted_pixel_features = pixel_features.permute(0, 2, 1)

#     for _ in range(n_iter):
#         dist_matrix = PairwiseDistFunction.apply(
#             pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

#         affinity_matrix = (-dist_matrix).softmax(1)
#         reshaped_affinity_matrix = affinity_matrix.reshape(-1)

#         mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
#         sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])
#         spixel_features = naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
#             / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)

#         spixel_features = spixel_features.permute(0, 2, 1)

#     hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

#     return sparse_abs_affinity, hard_labels, spixel_features


# def ssn_iter(pixel_features, num_spixels, n_iter):
#     """
#     computing assignment iterations
#     detailed process is in Algorithm 1, line 2 - 6

#     Args:
#         pixel_features: torch.Tensor
#             A Tensor of shape (B, C, H, W)
#         num_spixels: int
#             A number of superpixels
#         n_iter: int
#             A number of iterations
#         return_hard_label: bool
#             return hard assignment or not
#     """
#     height, width = pixel_features.shape[-2:]
#     num_spixels_width = int(math.sqrt(num_spixels * width / height))
#     num_spixels_height = int(math.sqrt(num_spixels * height / width))

#     spixel_features, init_label_map = \
#         calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
#     abs_indices = get_abs_indices(init_label_map, num_spixels_width)

#     pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
#     permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

#     for _ in range(n_iter):
#         dist_matrix = PairwiseDistFunction.apply(
#             pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

#         affinity_matrix = (-dist_matrix).softmax(1)
#         reshaped_affinity_matrix = affinity_matrix.reshape(-1)

#         mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
#         sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

#         abs_affinity = sparse_abs_affinity.to_dense().contiguous()
#         spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
#             / (abs_affinity.sum(2, keepdim=True) + 1e-16)

#         spixel_features = spixel_features.permute(0, 2, 1).contiguous()


#     hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

#     return abs_affinity, hard_labels, spixel_features


def soft_slic_all(point, seed,   n_iter=10,k_faces=8):
    """
    soft slic with all points entering computation

    Args:
        point (Tensor): import feature
        seed (Tensor): facet center
        n_iter (int, optional): number of ssn iter. Defaults to 10.
        k_facets (int, optional): Dummy API,never mind

    Returns:
        [type]: [description]
    """    
    for _ in range(n_iter):
        dist_matrix = point.new(
            point.shape[0], seed.shape[-1], point.shape[-1]).zero_()
        for i in range(seed.shape[-1]):
            initials = seed[:, :,i].unsqueeze(-1).repeat(1, 1, point.shape[-1])
            dist_matrix[:, i, :] = -(pairwise_distance(point,initials)*pairwise_distance(point, initials))
        QT = dist_matrix.softmax(1)
        seed = (torch.bmm(QT, point.permute(0, 2, 1)) /
                QT.sum(2, keepdim=True)).permute(0, 2, 1)
    _, hard_label = QT.permute(0, 2, 1).max(-1)
    return QT, hard_label, seed,point


def soft_slic_knn(point, seed, n_iter=10, k_facets=8):
    """
    soft slic with knn implemented by pytorch
    catch k adjacent spixs around selected center

    Args:
        point (Tensor): import feature
        seed (Tensor): facet center
        n_iter (int, optional): number of ssn iter. Defaults to 10.
        k_facets (int, optional): number of nearst facets knn chosed. Defaults to 4.
    """
    def soft_slic_all_single(point, seed):
        dist_matrix = point.new(
            point.shape[0], seed.shape[-1], point.shape[-1]).zero_()
        for i in range(seed.shape[-1]):
            initials = seed[:, :,
                            i].unsqueeze(-1).repeat(1, 1, point.shape[-1])
            dist_matrix[:, i, :] = -(pairwise_distance(point,
                                                       initials)*pairwise_distance(point, initials))
        QT = dist_matrix.softmax(1)
        _, hard_label = QT.permute(0, 2, 1).max(-1)
        return hard_label

    B, C, N = point.shape
    #select knn functions
    if point.device == 'cpu':
        knn = knn_indices_func_cpu
    else:
        knn = knn_indices_func_gpu

    #calculate initial superpixels
    # traditional k-means based
    #print("knn slic start")
    time1 = time()
    hard_label = soft_slic_all_single(point, seed)
    SoftSlicTime["single"] = time()-time1
    for iter in range(n_iter):
        time_iter = time()
        #time_ = time()
        dist_matrix = point.new(
            point.shape[0], seed.shape[-1], point.shape[-1]).zero_()
        NearstFacetsIdx = knn(seed, seed, k_facets)
        #batchwise operation->fvck
        for batch_idx in range(B):
            time_batch = time()
            for seed_i in range(seed.shape[-1]):
                mask = hard_label[batch_idx] == seed_i
                for _, nearst in enumerate(NearstFacetsIdx[batch_idx, seed_i]):
                    mask |= hard_label[batch_idx] == nearst
                pointt = point[batch_idx, :, mask].permute(1, 0)
                seeed = seed[batch_idx, :, seed_i].unsqueeze(
                    0).repeat(pointt.shape[0], 1)
                dist = pairwise_distance(pointt, seeed)
                Q_part = (-dist.pow(2)).softmax(-1)
                dist_matrix[batch_idx, seed_i, mask] = Q_part
            SoftSlicTime["time_batch_{}".format(
                batch_idx)] = time() - time_batch
        seed = (torch.bmm(dist_matrix, point.permute(0, 2, 1)) /
                dist_matrix.sum(2, keepdim=True)).permute(0, 2, 1)
        SoftSlicTime["time_iter_{}".format(iter)] = time()-time_iter
    _, hard_label = dist_matrix.permute(0, 2, 1).max(-1)
    SoftSlicTime["total"] = time()-time1
    return dist_matrix, hard_label, seed, point



def soft_slic_pknn(point, seed, n_iter=10, k_facets=256):
    """

    soft slic implemented using knn
    catch k adjacent pix around center

    Args:
        point (Tensor): point feature
        seed (Tensor): original spix center
        n_iter (int, optional): n slic iters . Defaults to 10.
        k_facets (int, optional): k nearst points. Defaults to 256.

    Returns:
        [type]: [description]
    """    
    B, C, N = point.shape
    #select knn functions
    if point.device == 'cpu':
        knn = knn_indices_func_cpu
    else:
        knn = knn_indices_func_gpu
    for _ in range(n_iter):
        dist_matrix = point.new(
            B, seed.shape[-1], point.shape[-1]).zero_()
        NearstFacetsIdx = knn(seed,point,k_point)
        # pointt = torch.stack([torch.stack([x[:, idxxx] for idxxx in idxx], dim=1) for idxx, x in zip(NearstFacetsIdx, point)])
        pointt = torch.stack([x[:,idxx] for idxx,x in zip(NearstFacetsIdx, point)])
        # print(pointt.shape)
        packed_seed = seed.unsqueeze(-1).expand([-1,-1,-1,k_point])
        dist =pairwise_distance(packed_seed,pointt)
        # print(dist.shape)
        QT_part = softmax(-dist.pow(2))
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                dist_matrix[i,j,NearstFacetsIdx[i,j]]=QT_part[i,j]
        seed = (torch.bmm(dist_matrix, point.permute(0, 2, 1))/(dist_matrix.sum(2, keepdim=True)+1e-16)).permute(0, 2, 1)
    _, hard_label = dist_matrix.permute(0, 2, 1).max(-1)

    return dist_matrix, hard_label, seed,point