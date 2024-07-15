# -*- coding: utf-8 -*-
# @Time   : 2022/3/23
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com
import math
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType


class DisenCDR(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DisenCDR, self).__init__(config, dataset)
        opt = config
        self.opt = config

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']
        self.embedding_size = config["feature_dim"]

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='csr', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='csr', value_field=None, domain='target').astype(np.float32)
        self.source_UV, self.source_VU = self.get_UV_and_VU_matrix(self.source_interaction_matrix, 'source')
        self.target_UV, self.target_VU = self.get_UV_and_VU_matrix(self.target_interaction_matrix, 'target')

        # define layers and loss
        self.source_specific_GNN = singleVBGE(opt, 0, self.overlapped_num_users, self.target_num_users, self.total_num_users, 0, self.target_num_items)
        self.source_share_GNN = singleVBGE(opt)

        self.target_specific_GNN = singleVBGE(opt, 0, self.overlapped_num_users, self.overlapped_num_users, self.target_num_users, self.target_num_items, self.total_num_items)
        self.target_share_GNN = singleVBGE(opt)

        opt['rate'] = self.get_rate(self.source_interaction_matrix, self.target_interaction_matrix)
        opt['target_num_users'] = self.target_num_users
        opt['overlapped_num_users'] = self.overlapped_num_users
        opt['total_num_users'] = self.total_num_users
        opt['target_num_items'] = self.target_num_items
        opt['overlapped_num_items'] = self.overlapped_num_items
        opt['total_num_items'] = self.total_num_items
        opt['device'] = self.device
        self.share_GNN = crossVBGE(opt)
        self.dropout = opt["dropout"]

        self.source_user_embedding = nn.Embedding(self.source_num_users - 1, opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(self.target_num_users - 1, opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(self.source_num_items - 1, opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(self.target_num_items - 1, opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(self.source_num_users - 1, opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(self.target_num_users - 1, opt["feature_dim"])

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        
        # storage variables for full sort evaluation acceleration
        self.source_restore_user_e = None
        self.source_restore_item_e = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        # self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['source_restore_user_e', 'source_restore_item_e', 'target_restore_user_e', 'target_restore_item_e']
        self.step = 0
        # with torch.no_grad():
        #     self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
        #     self.source_user_embedding_share.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
        #     self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

        #     self.target_user_embedding.weight[self.target_num_users:].fill_(0)
        #     self.target_user_embedding_share.weight[self.target_num_users:].fill_(0)
        #     self.target_item_embedding.weight[self.target_num_items:].fill_(0)

    def get_UV_and_VU_matrix(self, interaction_matrix, scheme='source'):
        interaction_matrix = interaction_matrix.toarray()
        if scheme == 'source':
            # get user
            interaction_matrix = np.concatenate([
                interaction_matrix[1:self.overlapped_num_users],
                interaction_matrix[self.target_num_users:]
            ])
            # no overlap item
            interaction_matrix = interaction_matrix[:, self.target_num_items:]
        elif scheme == 'target':
            # get user
            interaction_matrix = interaction_matrix[1:self.target_num_users]
            # no overlap item
            interaction_matrix = interaction_matrix[:, 1:self.target_num_items]

        interaction_matrix = sp.csr_matrix(interaction_matrix)
        UV = self.normalize(interaction_matrix)
        VU = self.normalize(interaction_matrix.T)
        UV = self.sparse_mx_to_torch_sparse_tensor(UV).to('cuda')
        VU = self.sparse_mx_to_torch_sparse_tensor(VU).to('cuda')
        return UV, VU

    def get_rate(self, source_interaction_matrix, target_interaction_matrix):
        source_user_interacted_num = source_interaction_matrix.sum(-1)
        target_user_interacted_num = target_interaction_matrix.sum(-1)
        source_rate = source_user_interacted_num / (source_user_interacted_num + target_user_interacted_num)
        source_rate = np.nan_to_num(source_rate)
        return source_rate

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
            # sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward_share(self):
        source_user = self.source_user_embedding.weight
        target_user = self.target_user_embedding.weight
        source_item = self.source_item_embedding.weight
        target_item = self.target_item_embedding.weight
        source_user_share = self.source_user_embedding_share.weight
        target_user_share = self.target_user_embedding_share.weight

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, self.source_UV, self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, self.target_UV, self.target_VU)

        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(source_user, self.source_UV, self.source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(target_user, self.target_UV, self.target_VU)

        mean, sigma, = self.share_GNN(source_user_share, target_user_share, self.source_UV, self.source_VU, self.target_UV, self.target_VU)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_user_mean_share = torch.cat([mean[1:self.overlapped_num_users], mean[self.target_num_users:]])
        source_user_sigma_share = torch.cat([sigma[1:self.overlapped_num_users], sigma[self.target_num_users:]])
        target_user_mean_share = mean[1:self.target_num_users]
        target_user_sigma_share = sigma[1:self.target_num_users]


        source_share_kld = self._kld_gauss(source_user_mean_share, source_user_sigma_share, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(target_user_mean_share, target_user_sigma_share, target_user_mean, target_user_sigma)

        self.kld_loss =  share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
            "beta"] * target_share_kld

        # source_learn_user = user_share + source_learn_specific_user
        # target_learn_user = user_share + target_learn_specific_user
        # Add embedding of overlapped users
        source_learn_user = torch.zeros_like(user_share, device=self.device)
        source_learn_user[1:self.overlapped_num_users] = user_share[1:self.overlapped_num_users] + source_learn_specific_user[:self.overlapped_num_users - 1]
        source_learn_user[self.target_num_users:] = source_learn_specific_user[self.overlapped_num_users - 1:]

        target_learn_user = torch.zeros_like(user_share, device=self.device)
        target_learn_user[1:self.overlapped_num_users] = user_share[1:self.overlapped_num_users] + target_learn_specific_user[:self.overlapped_num_users - 1]
        target_learn_user[self.overlapped_num_users:self.target_num_users] = target_learn_specific_user[self.overlapped_num_users - 1:]

        
        source_learn_specific_item = self.recbole_padding(source_learn_specific_item, 'item', 'source')
        target_learn_specific_item = self.recbole_padding(target_learn_specific_item, 'item', 'target')

        return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item

    def recbole_padding(self, embedding, ui='user', domain='source'):
        if domain == 'source':
            if ui == 'user':
                temp = torch.zeros(self.total_num_users, self.embedding_size, device=self.device)
                temp[1:self.overlapped_num_users] = embedding[:self.overlapped_num_users - 1]
                temp[self.target_num_users:] = embedding[self.overlapped_num_users - 1:]
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[self.target_num_items:] = embedding
        elif domain == 'target':
            if ui == 'user':
                temp = torch.zeros(self.total_num_users, self.embedding_size, device=self.device)
                temp[1:self.target_num_users] = embedding
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[1:self.target_num_items] = embedding
        return temp
    
    def forward_warmup(self):
        source_user = self.source_user_embedding.weight
        target_user = self.target_user_embedding.weight
        source_item = self.source_item_embedding.weight
        target_item = self.target_item_embedding.weight

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          self.source_UV, self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          self.target_UV, self.target_VU)
        self.kld_loss = 0
        source_learn_specific_user = self.recbole_padding(source_learn_specific_user, 'user', 'source')
        target_learn_specific_user = self.recbole_padding(target_learn_specific_user, 'user', 'target')
        source_learn_specific_item = self.recbole_padding(source_learn_specific_item, 'item', 'source')
        target_learn_specific_item = self.recbole_padding(target_learn_specific_item, 'item', 'target')
        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item

    def calculate_loss(self, interaction):
        self.init_restore_e()
        if self.step < 65:
            self.forward = self.forward_warmup
        else:
            self.forward = self.forward_share
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        losses = []

        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        # calculate BCE Loss in source domain
        source_output = torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1)
        source_bce_loss = self.loss(source_output, source_label)

        source_loss = source_bce_loss
        losses.append(source_loss)

        # calculate BCE Loss in target domain
        target_output = torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1)
        target_bce_loss = self.loss(target_output, target_label)

        target_loss = target_bce_loss

        losses.append(target_loss)


        kld_loss = self.source_specific_GNN.encoder[-1].kld_loss + self.target_specific_GNN.encoder[-1].kld_loss
        losses.append(kld_loss)
        if self.step >= 65:
            losses.append(self.kld_loss)
        else:
            self.step += 1

        return tuple(losses)

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]

        restore_user_e, restore_item_e, _, _ = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[self.target_num_items - self.overlapped_num_items:]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        _, _, restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.source_restore_user_e is not None or self.source_restore_item_e is not None:
            self.source_restore_user_e, self.source_restore_item_e = None, None
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.source_restore_user_e is None or self.source_restore_item_e is None or self.target_restore_user_e is None or self.target_restore_item_e is None:
            self.source_restore_user_e, self.source_restore_item_e, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.source_restore_user_e, self.source_restore_item_e, self.target_restore_user_e, self.target_restore_item_e

# Base GCN
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = self.glorot_init(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x

# Single VBGE
class singleVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt, l1=None, r1=None, l2=None, r2=None, item_l=None, item_r=None):
        super(singleVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(SingleDGCNLayer(opt))
        self.encoder.append(SingleLastLayer(opt, l1, r1, l2, r2, item_l, item_r))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
        return mean, sigma

class SingleDGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(SingleDGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)


class SingleLastLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt, l1=None, r1=None, l2=None, r2=None, item_l=None, item_r=None):
        super(SingleLastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.l1, self.r1, self.l2, self.r2 = l1, r1, l2, r2
        self.item_l = item_l
        self.item_r = item_r
        self.user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters_user(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
            # sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        # mean = torch.cat([mean[self.l1:self.r1], mean[self.l2:self.r2]])
        # logstd = torch.cat([logstd[self.l1:self.r1], logstd[self.l2:self.r2]])
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def reparameters_item(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
            # sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        # mean = mean[self.item_l : self.item_r]
        # logstd = logstd[self.item_l : self.item_r]
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj,VU_adj)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = user_kld + item_kld

        return user, item


    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        user, kld_loss = self.reparameters_user(User_ho_mean, User_ho_logstd)
        return user, kld_loss

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        item, kld_loss = self.reparameters_item(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        # user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return User_ho_mean, User_ho_logstd
        # return user, kld_loss

# Cross VBGE
class crossVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        learn_user_source = source_ufea
        learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
            learn_user_source = torch.cat([
                learn_user_source[1:layer.overlapped_num_users],
                learn_user_source[layer.target_num_users:]
            ])
            learn_user_target = learn_user_target[1:layer.target_num_users]

        mean, sigma, = self.encoder[-1](learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
        return mean, sigma

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.embedding_size = opt["feature_dim"]
        self.device = opt["device"]
        self.target_num_users = opt['target_num_users']
        self.overlapped_num_users = opt['overlapped_num_users']
        self.total_num_users = opt['total_num_users']
        self.target_num_items = opt['target_num_items']
        self.overlapped_num_items = opt['overlapped_num_items']
        self.total_num_items = opt['total_num_items']
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1).unsqueeze(-1)
        self.source_rate = self.source_rate.cuda()

    
    def recbole_padding(self, embedding, ui='user', domain='source'):
        if domain == 'source':
            if ui == 'user':
                temp = torch.zeros(self.opt['total_num_users'], self.embedding_size, device=self.device)
                temp[1:self.overlapped_num_users] = embedding[:self.overlapped_num_users - 1]
                temp[self.target_num_users:] = embedding[self.overlapped_num_users - 1:]
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[self.target_num_items:] = embedding
        elif domain == 'target':
            if ui == 'user':
                temp = torch.zeros(self.opt['total_num_users'], self.embedding_size, device=self.device)
                temp[1:self.target_num_users] = embedding
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[1:self.target_num_items] = embedding
        return temp

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)

        source_User = torch.cat((source_User_ho , source_ufea), dim=1)
        source_User = self.source_user_union(source_User)
        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)
        
        source_User = self.recbole_padding(source_User, 'user', 'source')
        target_User = self.recbole_padding(target_User, 'user', 'target')

        return self.source_rate * F.relu(source_User) +  (1 - self.source_rate) * F.relu(target_User), self.source_rate * F.relu(source_User) + (1 - self.source_rate) * F.relu(target_User)


class LastLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.embedding_size = opt["feature_dim"]
        self.device = opt["device"]
        
        self.target_num_users = opt['target_num_users']
        self.overlapped_num_users = opt['overlapped_num_users']
        self.total_num_users = opt['total_num_users']
        self.target_num_items = opt['target_num_items']
        self.overlapped_num_items = opt['overlapped_num_items']
        self.total_num_items = opt['total_num_items']
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.source_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1).unsqueeze(-1)

        self.source_rate = self.source_rate.cuda()


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            # sampled_z = gaussian_noise * sigma + mean
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def recbole_padding(self, embedding, ui='user', domain='source'):
        if domain == 'source':
            if ui == 'user':
                temp = torch.zeros(self.opt['total_num_users'], self.embedding_size, device=self.device)
                temp[1:self.overlapped_num_users] = embedding[:self.overlapped_num_users - 1]
                temp[self.target_num_users:] = embedding[self.overlapped_num_users - 1:]
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[self.target_num_items:] = embedding
        elif domain == 'target':
            if ui == 'user':
                temp = torch.zeros(self.opt['total_num_users'], self.embedding_size, device=self.device)
                temp[1:self.target_num_users] = embedding
            elif ui == 'item':
                temp = torch.zeros(self.total_num_items, self.embedding_size, device=self.device)
                temp[1:self.target_num_items] = embedding
        return temp
    
    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        source_User_mean = torch.cat(
            (source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)

        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)

        target_User_mean = torch.cat(
            (target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)

        target_User_logstd = torch.cat(
            (target_User_ho_logstd, target_ufea),
            dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)

        source_User_mean = self.recbole_padding(source_User_mean, 'user', 'source')
        target_User_mean = self.recbole_padding(target_User_mean, 'user', 'target')
        source_User_logstd = self.recbole_padding(source_User_logstd, 'user', 'source')
        target_User_logstd = self.recbole_padding(target_User_logstd, 'user', 'target')

        return self.source_rate * source_User_mean + (1 - self.source_rate) * target_User_mean, self.source_rate * source_User_logstd + (1 - self.source_rate) * target_User_logstd
