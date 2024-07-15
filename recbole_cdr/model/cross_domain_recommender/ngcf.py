

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.utils import InputType


class NGCF(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['latent_dim']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.latent_dim] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.interaction_matrix = (self.source_interaction_matrix + self.target_interaction_matrix).tocoo()
        self.norm_adj_matrix = self.get_norm_adj_mat(self.interaction_matrix, self.total_num_users, self.total_num_items).to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)
        
        # storage variables for full sort evaluation acceleration
        self.source_restore_user_e = None
        self.source_restore_item_e = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['source_restore_user_e', 'source_restore_item_e', 'target_restore_user_e', 'target_restore_item_e']

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.total_num_items + self.total_num_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def graph_layer(self, adj_matrix, all_embeddings):
        embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        return embeddings
    def forward(self):
        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.total_num_users, self.total_num_items]
        )

        return user_all_embeddings, item_all_embeddings, user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        source_user = interaction[self.SOURCE_USER_ID]
        source_pos = interaction[self.SOURCE_ITEM_ID]
        source_neg = interaction[self.SOURCE_NEG_ITEM_ID]

        target_user = interaction[self.TARGET_USER_ID]
        target_pos = interaction[self.TARGET_ITEM_ID]
        target_neg = interaction[self.TARGET_NEG_ITEM_ID]
        losses = []

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_pos_embeddings = source_item_all_embeddings[source_pos]
        source_i_neg_embeddings = source_item_all_embeddings[source_neg]

        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_pos_embeddings = target_item_all_embeddings[target_pos]
        target_i_neg_embeddings = target_item_all_embeddings[target_neg]

        # calculate BPR Loss in source domain
        pos_scores = torch.mul(source_u_embeddings, source_i_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(source_u_embeddings, source_i_neg_embeddings).sum(dim=1)
        source_bpr_loss = self.loss(pos_scores, neg_scores)

        # calculate Reg Loss in source domain
        u_ego_embeddings = self.user_embedding(source_user)
        i_pos_ego_embeddings = self.item_embedding(source_pos)
        i_neg_ego_embeddings = self.item_embedding(source_neg)
        source_reg_loss = self.reg_loss(u_ego_embeddings, i_pos_ego_embeddings, i_neg_ego_embeddings)

        source_loss = source_bpr_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)

        # calculate BPR Loss in target domain
        pos_scores = torch.mul(target_u_embeddings, target_i_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(target_u_embeddings, target_i_neg_embeddings).sum(dim=1)
        target_bpr_loss = self.loss(pos_scores, neg_scores)

        # calculate Reg Loss in target domain
        u_ego_embeddings = self.user_embedding(target_user)
        i_pos_ego_embeddings = self.item_embedding(target_pos)
        i_neg_ego_embeddings = self.item_embedding(target_neg)
        target_reg_loss = self.reg_loss(u_ego_embeddings,i_pos_ego_embeddings, i_neg_ego_embeddings)

        target_loss = target_bpr_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)

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
