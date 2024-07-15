
import torch
import torch.nn as nn

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers


class PLE(CrossDomainRecommender):
    r"""CoNet takes neural network as the basic model and uses cross connections
        unit to improve the learning of matching functions in the current domain.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(PLE, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.bottom_mlp_dims = config["bottom_mlp_dims"]  # list type: the list of hidden layers size
        self.tower_mlp_dims = config["tower_mlp_dims"]  # list type: the list of hidden layers size
        self.n_experts = config['n_experts']
        self.n_experts_spe = config['n_experts_specific']
        self.drop_rate = config['drop_rate']

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        self.loss = nn.BCELoss()

        self.expert = torch.nn.ModuleList([
            MLPLayers([self.latent_dim * 2] + self.bottom_mlp_dims, self.drop_rate) for i in range(self.n_experts)
        ])
        self.expert_source = torch.nn.ModuleList([
            MLPLayers([self.latent_dim * 2] + self.bottom_mlp_dims, self.drop_rate) for i in range(self.n_experts_spe)
        ])
        self.expert_target = torch.nn.ModuleList([
            MLPLayers([self.latent_dim * 2] + self.bottom_mlp_dims, self.drop_rate) for i in range(self.n_experts_spe)
        ])
        self.tower_source = MLPLayers([self.bottom_mlp_dims[-1]] + self.tower_mlp_dims, self.drop_rate)
        self.tower_target = MLPLayers([self.bottom_mlp_dims[-1]] + self.tower_mlp_dims, self.drop_rate)
        self.gate_source = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim * 2, self.n_experts_spe + self.n_experts),
            torch.nn.Softmax(dim=1)
        )
        self.gate_target = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim * 2, self.n_experts_spe + self.n_experts),
            torch.nn.Softmax(dim=1)
        )
        self.predict_layer_source = nn.Sequential(
            nn.Linear(self.tower_mlp_dims[-1], 1),
            nn.Sigmoid(),
        )
        self.predict_layer_target = nn.Sequential(
            nn.Linear(self.tower_mlp_dims[-1], 1),
            nn.Sigmoid(),
        )

        # parameters initialization
        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)


    def source_forward(self, user, item, input=None):
        if input == None:
            source_user_embedding = self.source_user_embedding(user)
            source_item_embedding = self.source_item_embedding(item)
            source_input = torch.cat([source_user_embedding, source_item_embedding], dim=1).to(self.device)
        else:
            source_input = input
        share_out = torch.cat([self.expert[i](source_input).unsqueeze(1) for i in range(self.n_experts)], dim = 1)
        task_out = torch.cat([self.expert_source[i](source_input).unsqueeze(1) for i in range(self.n_experts_spe)], dim = 1)
        mix_out = torch.cat([task_out, share_out], dim=1)
        gate_source = self.gate_source(source_input).unsqueeze(1)
        task_fea = torch.bmm(gate_source, mix_out).squeeze(1)
        task_out = self.tower_source(task_fea)
        task_out = self.predict_layer_source(task_out).squeeze(1)
        return task_out

    def target_forward(self, user, item, input=None):
        if input == None:
            target_user_embedding = self.target_user_embedding(user)
            target_item_embedding = self.target_item_embedding(item)
            target_input = torch.cat([target_user_embedding, target_item_embedding], dim=1).to(self.device)
        else:
            target_input = input
        share_out = torch.cat([self.expert[i](target_input).unsqueeze(1) for i in range(self.n_experts)], dim = 1)
        task_out = torch.cat([self.expert_target[i](target_input).unsqueeze(1) for i in range(self.n_experts_spe)], dim = 1)
        mix_out = torch.cat([task_out, share_out], dim=1)
        gate_target = self.gate_target(target_input).unsqueeze(1)
        task_fea = torch.bmm(gate_target, mix_out).squeeze(1)
        task_out = self.tower_target(task_fea)
        task_out = self.predict_layer_target(task_out).squeeze(1)
        return task_out

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_source = self.source_forward(source_user, source_item)
        p_target = self.target_forward(target_user, target_item)

        loss_s = self.loss(p_source, source_label)
        loss_t = self.loss(p_target, target_label)

        return tuple([loss_s, loss_t])

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        user_e = self.source_user_embedding(user)
        user_num = user_e.shape[0]
        all_item_e = self.source_item_embedding.weight[self.target_num_items - self.overlapped_num_items:]
        item_num = all_item_e.shape[0]
        all_user_e = user_e.repeat(1, item_num).view(-1, self.latent_dim)
        user_e_list = torch.split(all_user_e, [item_num]*user_num)
        score_list = []
        for u_embed in user_e_list:
            input = torch.cat([u_embed, all_item_e], dim=1)
            score_list.append(self.source_forward(None, None, input).unsqueeze(-1))
        score = torch.cat(score_list, dim=1).transpose(0, 1)
        return score

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        user_e = self.target_user_embedding(user)
        user_num = user_e.shape[0]
        all_item_e = self.target_item_embedding.weight[:self.target_num_items]
        item_num = all_item_e.shape[0]
        all_user_e = user_e.repeat(1, item_num).view(-1, self.latent_dim)
        user_e_list = torch.split(all_user_e, [item_num]*user_num)
        score_list = []
        for u_embed in user_e_list:
            input = torch.cat([u_embed, all_item_e], dim=1)
            score_list.append(self.target_forward(None, None, input).unsqueeze(-1))
        score = torch.cat(score_list, dim=1).transpose(0, 1)
        return score
