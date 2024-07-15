import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender, AutoEncoderMixin
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

from recbole_cdr.model.layers import Binarize


class MDAP(CrossDomainRecommender, AutoEncoderMixin):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MDAP, self).__init__(config, dataset)

        # Load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.convert_sparse_matrix_to_rating_matrix(self.source_interaction_matrix + self.target_interaction_matrix)
        
        # Load parameters info
        self.device = config['device']
        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]
        self.drop_out = config["dropout_prob"]
        self.kfac = config["kfac"]
        self.tau = config["tau"]
        self.tau_source = config["tau_source"]
        self.tau_target = config["tau_target"]
        self.nogb = config["nogb"]
        self.nobinarize = config["nobinarize"]
        self.regs = config["reg_weights"]

        # Define layers and loss
        self.update = 0
        self.encode_layer_dims = self.layers + [self.lat_dim]
        
        self.encoder_source = MLPLayers([self.source_num_items] + self.encode_layer_dims, activation="tanh")
        self.encoder_target = MLPLayers([self.target_num_items] + self.encode_layer_dims, activation="tanh")
        self.item_embedding = nn.Embedding(self.total_num_items, self.lat_dim)
        self.k_embedding_source = nn.Embedding(self.kfac, self.lat_dim)
        self.k_embedding_target = nn.Embedding(self.kfac, self.lat_dim)
        self.gate_embedding = nn.Embedding(2, embedding_dim=self.kfac)
        self.l2_loss = EmbLoss()

        # Additional layers
        self.item_embedding_source = nn.Embedding(self.source_num_items, self.lat_dim)
        self.item_embedding_target = nn.Embedding(self.target_num_items, self.lat_dim)
        self.mask_source = nn.Parameter(torch.empty(1, self.kfac))
        self.mask_target = nn.Parameter(torch.empty(1, self.kfac))
        self.binary = Binarize.apply

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims) - 2:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def get_sparse_norm_rating_matrix(self):
        rating_matrix = F.normalize(self.rating_matrix).to_sparse()
        self.sparse_norm_rating_matrix = rating_matrix.to(self.device)
        rating_matrix = torch.tensor(self.source_interaction_matrix.toarray())
        self.sparse_norm_rating_source = F.normalize(rating_matrix).to_sparse().to(self.device)
        rating_matrix = torch.tensor(self.target_interaction_matrix.toarray())
        self.sparse_norm_rating_target = F.normalize(rating_matrix).to_sparse().to(self.device)

        self.rating_matrix_source = torch.cat([
            self.rating_matrix[:, :self.overlapped_num_items],
            self.rating_matrix[:, self.target_num_items:]
        ], dim=-1)
        self.rating_matrix_target = self.rating_matrix[:self.target_num_users][:, :self.target_num_items]

    def get_cates(self, cates_logits):
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        elif self.nobinarize:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode
        else:
            cates = self.binary(cates_logits)
        return cates

    def forward(self, rating_matrix_source, rating_matrix_target):
        # Normalize item embeddings
        items_source = F.normalize(self.item_embedding_source.weight, dim=1)
        items_target = F.normalize(self.item_embedding_target.weight, dim=1)
        
        # Normalize and apply softmax to gate embeddings
        gates = F.normalize(self.gate_embedding.weight, dim=1)
        gates = torch.softmax(gates, dim=-1)
        
        # Normalize core embeddings
        cores_source = F.normalize(self.k_embedding_source.weight, dim=1)
        cores_target = F.normalize(self.k_embedding_source.weight, dim=1)
        
        # Normalize and apply dropout to rating matrices
        rating_matrix_source = F.normalize(rating_matrix_source)
        rating_matrix_source = F.dropout(rating_matrix_source, self.drop_out, training=self.training)
        rating_matrix_target = F.normalize(rating_matrix_target)
        rating_matrix_target = F.dropout(rating_matrix_target, self.drop_out, training=self.training)
        
        # Compute category logits
        cates_logits_source = torch.matmul(items_source, cores_source.transpose(0, 1)) / self.tau_source
        cates_logits_target = torch.matmul(items_target, cores_target.transpose(0, 1)) / self.tau_source
        
        # Get category assignments
        cates_source = self.get_cates(cates_logits_source)
        cates_target = self.get_cates(cates_logits_target)
        
        probs_source = None
        probs_target = None

        for k in range(self.kfac):
            # Process source domain
            cates_k = cates_source[:, k].reshape(1, -1)
            x_k = rating_matrix_source * cates_k
            h = self.encoder_source(x_k)
            z = h
            z_k = F.normalize(z, dim=1)
            logits_k = torch.matmul(z_k, items_source.transpose(0, 1)) / self.tau_source
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            gate_s_k = gates[0, k]
            #gate control
            probs_k = probs_k.mul(gate_s_k)
            probs_source = probs_k if probs_source is None else probs_source + probs_k

            # Process target domain
            cates_k = cates_target[:, k].reshape(1, -1)
            x_k = rating_matrix_target * cates_k
            h = self.encoder_target(x_k)
            z = h
            logits_k = torch.matmul(z, items_target.transpose(0, 1)) / self.tau_target
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            gate_t_k = gates[1, k]
            #gate control
            probs_k = probs_k.mul(gate_t_k)
            probs_target = probs_k if probs_target is None else probs_target + probs_k

        # Compute final logits
        logits_source = torch.log(probs_source + 1e-12)
        logits_target = torch.log(probs_target + 1e-12)

        return logits_source, logits_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]
        source_user_id = self.process_source_user_id(source_user_id)
        rating_matrix = self.rating_matrix[torch.cat([source_user_id, target_user_id]).cpu()].to(source_user_id.device)

        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(self.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(self.device)

        z_source, z_target = self.forward(rating_matrix_source, rating_matrix_target)

        ce_loss_source = -(F.log_softmax(z_source, 1) * rating_matrix_source).sum(1).mean()
        ce_loss_target = -(F.log_softmax(z_target, 1) * rating_matrix_target).sum(1).mean()
        gates = F.normalize(self.gate_embedding.weight, dim=1)
        gates = torch.softmax(gates, dim=-1)
        loss_4 = self.regs[2] * (gates[0, :] * gates[1, :]).sum()

        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss_source + ce_loss_target + self.reg_loss() + loss_4
        return ce_loss_source + ce_loss_target + loss_4

    def reg_loss(self):
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding.weight.norm(2)
        loss_2 = reg_1 * self.k_embedding_source.weight.norm(2)
        loss_3 = 0
        
        for name, parm in self.encoder_source.named_parameters():
            if name.endswith("weight"):
                loss_3 += reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        rating_matrix = self.rating_matrix_source[user.cpu()].to(self.device)
        scores, _ = self.forward(rating_matrix, self.rating_matrix_target[[0, 1]].to(self.device))
        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        rating_matrix = self.rating_matrix_target[user.cpu()].to(user.device)
        _, scores = self.forward(self.rating_matrix_source[[0, 1]].to(self.device), rating_matrix)
        return scores.reshape(-1)