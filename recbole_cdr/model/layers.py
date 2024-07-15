import math
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
import torch.nn as nn
from torch_geometric.utils import degree
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers

BIG_CONSTANT = 1e8

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real_value_mask):
        binarized_mask = (real_value_mask >= 0).float().requires_grad_(True)
        ctx.save_for_backward(binarized_mask.clone())
        return binarized_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[0] ** -0.5)

    data_dash = torch.einsum("nhd,md->nhm", (data_normalizer * data), projection_matrix) # perform projection
    diag_data = (data ** 2).sum(-1)

    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )
    return data_dash

class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_factor,
                    nb_random_features=10, weight=True):
        super(KernelAttention, self).__init__()
        if weight:
            self.Wk = nn.ModuleList()
            self.Wq = nn.ModuleList()
            self.Wv = nn.ModuleList()
            for _ in range(num_factor):
                self.Wk.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wq.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wv.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))

        self.out_channels = out_channels
        self.num_factor = num_factor
        self.nb_random_features = nb_random_features
        self.weight = weight

    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

    def forward(self, z, tau):
        query, key, value = torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device)
        for head in range(self.num_factor):
            query[:, head] = self.Wq[head](z[:, head])
            key[:, head] = self.Wk[head](z[:, head])
            value[:, head] = self.Wv[head](z[:, head])

        dim = query.shape[-1]
        projection_matrix = create_projection_matrix(self.nb_random_features, dim).to(query.device)
        z_next = kernelized_softmax(query, key, value, projection_matrix, tau)

        z_next = z_next.flatten(-2, -1)
        return z_next.squeeze()

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

def create_projection_matrix(m, d):
    nb_full_blocks = int(m/d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.T)
    final_matrix = torch.vstack(block_list)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)

def kernelized_softmax(query, key, value, projection_matrix=None, tau=0.5):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    query_kernel = softmax_kernel(query, True, projection_matrix)
    key_kernel = softmax_kernel(key, False, projection_matrix)

    kvs = torch.einsum("nhm,nhd->hmd", key_kernel, value)
    numerator = torch.einsum("nhm,hmd->nhd", query_kernel, kvs)
    denominator = (query_kernel * key_kernel.sum(0, keepdim=True)).sum(-1, keepdim=True)

    z_output = numerator / denominator
    return z_output