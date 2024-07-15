        

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType


class BPR(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.alpha = config['alpha']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.source_restore_user_e = None
        self.source_restore_item_e = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['source_restore_user_e', 'source_restore_item_e', 'target_restore_user_e', 'target_restore_item_e']

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.total_num_users, self.total_num_items]
        )
        source_user_all_embeddings = user_all_embeddings
        source_item_all_embeddings = item_all_embeddings

        target_user_all_embeddings = user_all_embeddings
        target_item_all_embeddings = item_all_embeddings

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

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
        losses.append(self.alpha * source_loss)

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
        losses.append((1 - self.alpha) * target_loss)

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
