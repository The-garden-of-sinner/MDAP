
import dgl
from dgl.nn.pytorch.conv import GraphConv, SAGEConv
from dgl.transforms import DropNode, DropEdge

import numpy as np
import scipy.sparse as sp
from random import choice

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType
from recbole.model.layers import MLPLayers

def InfoNCE(view1, view2, temperature=0.2):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

class DRMTCDR(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DRMTCDR, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.n_sampling = config['n_sampling']
        self.tau = config['tau']
        self.cl_rate = config['cl_rate']
        self.drop_rate = config['drop_rate']
        self.big_dataset = config['big_dataset']

        # define layers and loss
        self.coarse_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.coarse_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        self.homo_mlp_user = nn.Parameter(data=torch.zeros(3, self.latent_dim // 3, self.latent_dim // 3))
        self.homo_mlp_item = nn.Linear(self.latent_dim, self.latent_dim)
        self.source_adapt_mlp = nn.Linear(self.latent_dim // 3 * 2, self.latent_dim)
        self.target_adapt_mlp = nn.Linear(self.latent_dim // 3 * 2, self.latent_dim)
        self.source_predict_user_mlp = MLPLayers([2 * (self.latent_dim + self.latent_dim // 3), 64], self.drop_rate)
        self.target_predict_user_mlp = MLPLayers([2 * (self.latent_dim + self.latent_dim // 3), 64], self.drop_rate)
        self.source_predict_item_mlp = MLPLayers([3 * self.latent_dim, 64], self.drop_rate)
        self.target_predict_item_mlp = MLPLayers([3 * self.latent_dim, 64], self.drop_rate)

        self.lightgcn_conv = GraphConv(1, 1, weight=False, bias=False, allow_zero_in_degree=True)
        self.graphsage_conv = SAGEConv(self.latent_dim, self.latent_dim, 'mean', bias=False)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.interaction_matrix = (self.source_interaction_matrix + self.target_interaction_matrix).tocoo()
        self.build_graph()
        
        # storage variables for full sort evaluation acceleration
        self.source_restore_user_e = None
        self.source_restore_item_e = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        nn.init.xavier_normal_(self.homo_mlp_user.data)
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        self.other_parameter_name = ['source_restore_user_e', 'source_restore_item_e', 'target_restore_user_e', 'target_restore_item_e']

    def expand_interaction_matrix(self, interaction_matrix):
        n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        A = sp.coo_matrix(A)
        return A

    def build_graph(self):
        # build coarse graph
        A = self.expand_interaction_matrix(self.interaction_matrix)
        self.coarse_graph = dgl.from_scipy(A).to(self.device)

        threshold = 0
        # build user concurrence graph
        interaction_matrix = self.interaction_matrix
        user_user_graph = (interaction_matrix @ interaction_matrix.T)
        mask = user_user_graph.data < threshold
        user_user_graph.data[mask] = 0
        user_user_graph.eliminate_zeros() # Reduce
        self.user_user_graph = dgl.from_scipy(user_user_graph).to(self.device)

        # build item concurrence graph
        interaction_matrix = self.interaction_matrix
        item_item_graph = interaction_matrix.T @ interaction_matrix
        mask = item_item_graph.data < threshold
        item_item_graph.data[mask] = 0
        item_item_graph.eliminate_zeros() # Reduce
        self.item_item_graph = dgl.from_scipy(item_item_graph).to(self.device)

        # build local graph
        source_graph = self.expand_interaction_matrix(self.source_interaction_matrix)
        self.source_graph = dgl.from_scipy(source_graph).to(self.device)
        target_graph = self.expand_interaction_matrix(self.target_interaction_matrix)
        self.target_graph = dgl.from_scipy(target_graph).to(self.device)

    def norm_adj_tensor(graph : torch.Tensor):
        graph = graph.coalesce()
        idx, val = graph.indices(), graph.values()
        degree = torch.sparse.sum(graph, dim=0).to_dense().pow(-0.5)
        degree = degree.nan_to_num(nan=0, posinf=0, neginf=0)
        val = val * degree[idx[0]] * degree[idx[1]]
        norm_graph = torch.sparse_coo_tensor(idx, val, size=graph.size(), device=graph.device)
        return norm_graph

    def scipy2tensor(self, graph) -> torch.Tensor:
        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = graph.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def get_augmented_graph(self, n_samples):
        g = self.coarse_graph
        g.ndata['nid'] = torch.arange(self.total_num_users + self.total_num_items, device=self.device)
        emb = self.get_ego_embeddings('coarse')
        original_graph_emb = F.normalize(self.graphsage_conv(g, emb).mean(0).unsqueeze(0))
        best_score, best_g = 0, None
        transform_list = [DropEdge(0.2), DropNode(0.2)]
        for _ in range(n_samples):
            aug = choice(transform_list)
            g_new = aug(g.clone())
            aug_graph_emb = F.normalize(self.graphsage_conv(g_new, emb[g_new.ndata['nid']]).mean(0).unsqueeze(0))
            score = original_graph_emb @ aug_graph_emb.T
            if score > best_score:
                best_g = g_new

        nid = best_g.ndata['nid']
        selected_user, selected_item = nid[nid < self.total_num_users], nid[nid >= self.total_num_users]
        adj = best_g.adj(scipy_fmt='csr')
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        norm_adj = norm_adj[:len(selected_user), len(selected_user):]
        self.adj_user = self.scipy2tensor(norm_adj).to('cuda')
        self.adj_item = self.scipy2tensor(norm_adj.T).to('cuda')

        return selected_user, selected_item

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
        elif domain == 'target':
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
        elif domain == 'coarse':
            user_embeddings = self.coarse_user_embedding.weight
            item_embeddings = self.coarse_item_embedding.weight

        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def lightgcn_layer(self, layer, graph, embedding):
        embeddings_list = [embedding]
        for layer_idx in range(layer):
            embedding = self.lightgcn_conv(graph, embedding)
            embeddings_list.append(embedding)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings

    def lightgcn_layer_scipy(self, layer, adj, embedding):
        embeddings_list = [embedding]
        for layer_idx in range(layer):
            embedding = torch.sparse.mm(adj.T, embedding)
            embedding = torch.sparse.mm(adj, embedding)
            embeddings_list.append(embedding)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings

    def forward(self):
        # Coarse Heterogeneous Convolution
        coarse_all_embeddings = self.get_ego_embeddings('coarse')
        coarse_all_embeddings = self.lightgcn_layer(self.n_layers, self.coarse_graph, coarse_all_embeddings)
        coarse_user_all_embeddings, coarse_item_all_embeddings = torch.split(
            coarse_all_embeddings, [self.total_num_users, self.total_num_items]
        )

        # Fine Homogenous Convolution
        homo_item_embeddings = self.lightgcn_layer(
            self.n_layers,
            self.item_item_graph,
            F.normalize(F.relu(self.homo_mlp_item(coarse_item_all_embeddings)))
        )

        homo_user_embeddings = coarse_user_all_embeddings.reshape(-1, 3, self.latent_dim // 3)
        homo_user_embeddings = torch.einsum('abc,bcf->abf', homo_user_embeddings, self.homo_mlp_user)
        homo_user_embeddings = self.lightgcn_layer(
            self.n_layers,
            self.user_user_graph,
            F.normalize(F.relu(homo_user_embeddings))
        )

        # get augmented embs
        if self.training:
            selected_user, selected_item = self.get_augmented_graph(self.n_sampling)
            aug_homo_item_embeddings = self.lightgcn_layer_scipy(
                self.n_layers,
                self.adj_item,
                F.normalize(F.relu(self.homo_mlp_item(coarse_item_all_embeddings)))[selected_item - self.total_num_users]
            )
            aug_item_embeddings = torch.zeros_like(homo_item_embeddings, device=self.device)
            aug_item_embeddings[selected_item - self.total_num_users] = aug_homo_item_embeddings

            aug_homo_user_embeddings = coarse_user_all_embeddings.reshape(-1, 3, self.latent_dim // 3)
            aug_homo_user_embeddings = torch.einsum('abc,bcf->abf', aug_homo_user_embeddings, self.homo_mlp_user)
            aug_homo_user_embeddings = self.lightgcn_layer_scipy(
                self.n_layers,
                self.adj_user,
                F.normalize(F.relu(aug_homo_user_embeddings))[selected_user].flatten(-2, -1)
            )
            aug_homo_user_embeddings = aug_homo_user_embeddings.reshape(-1, 3, self.latent_dim // 3)
            aug_user_embeddings = torch.zeros_like(homo_user_embeddings, device=self.device)
            aug_user_embeddings[selected_user] = aug_homo_user_embeddings

        # local domain adaptation
        source_adapt_user_embeddings = torch.cat([homo_user_embeddings[:, 0], homo_user_embeddings[:, -1]], dim=-1)
        source_adapt_user_embeddings = self.source_adapt_mlp(source_adapt_user_embeddings)
        source_adapt_all_embeddings = torch.cat([
            source_adapt_user_embeddings,
            homo_item_embeddings,
        ])
        source_adapt_all_embeddings = self.lightgcn_layer(
            1,
            self.source_graph,
            source_adapt_all_embeddings,
        )
        source_adapt_user_embeddings, source_adapt_item_embeddings = torch.split(
            source_adapt_all_embeddings, [self.total_num_users, self.total_num_items]
        )

        target_adapt_user_embeddings = torch.cat([homo_user_embeddings[:, 1], homo_user_embeddings[:, -1]], dim=-1)
        target_adapt_user_embeddings = self.target_adapt_mlp(target_adapt_user_embeddings)
        target_adapt_all_embeddings = torch.cat([
            target_adapt_user_embeddings,
            homo_item_embeddings,
        ])
        target_adapt_all_embeddings = self.lightgcn_layer(
            1,
            self.target_graph,
            target_adapt_all_embeddings,
        )
        target_adapt_user_embeddings, target_adapt_item_embeddings = torch.split(
            target_adapt_all_embeddings, [self.total_num_users, self.total_num_items]
        )
        

        # local domain prediction
        source_user_all_embeddings = self.source_predict_user_mlp(torch.cat([
            self.source_user_embedding.weight,
            source_adapt_user_embeddings,
            homo_user_embeddings[:, 0],
            homo_user_embeddings[:, -1]
        ], dim=-1))
        source_item_all_embeddings = self.source_predict_item_mlp(torch.cat([
            self.source_item_embedding.weight,
            source_adapt_item_embeddings,
            homo_item_embeddings
        ], dim=-1))

        target_user_all_embeddings = self.target_predict_user_mlp(torch.cat([
            self.target_user_embedding.weight,
            target_adapt_user_embeddings,
            homo_user_embeddings[:, 1],
            homo_user_embeddings[:, -1]
        ], dim=-1))
        target_item_all_embeddings = self.target_predict_item_mlp(torch.cat([
            self.target_item_embedding.weight,
            target_adapt_item_embeddings,
            homo_item_embeddings
        ], dim=-1))
        
        if self.training:
            return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings, \
                homo_user_embeddings, homo_item_embeddings, aug_user_embeddings, aug_item_embeddings, selected_user, selected_item
        else:
            return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings, \
            homo_user_embeddings, homo_item_embeddings, aug_homo_user_embeddings, aug_homo_item_embeddings, selected_user, selected_item = self.forward()
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
        losses.append(source_bpr_loss)

        # calculate BPR Loss in target domain
        pos_scores = torch.mul(target_u_embeddings, target_i_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(target_u_embeddings, target_i_neg_embeddings).sum(dim=1)
        target_bpr_loss = self.loss(pos_scores, neg_scores)
        losses.append(target_bpr_loss)

        # calculate contrastive loss
        loss_con = 0
        homo_source_user_embeddings = F.normalize(homo_user_embeddings[:, 0][source_user])
        homo_target_user_embeddings = F.normalize(homo_user_embeddings[:, 1][target_user])
        aug_homo_source_user_embeddings = F.normalize(aug_homo_user_embeddings[:, 0][source_user])
        aug_homo_target_user_embeddings = F.normalize(aug_homo_user_embeddings[:, 1][target_user])
        loss_con += InfoNCE(homo_source_user_embeddings, aug_homo_source_user_embeddings, self.tau)
        loss_con += InfoNCE(homo_target_user_embeddings, aug_homo_target_user_embeddings, self.tau)

        homo_source_user_embeddings = F.normalize(homo_user_embeddings[source_user] / self.tau)
        homo_target_user_embeddings = F.normalize(homo_user_embeddings[target_user] / self.tau)
        aug_homo_source_user_embeddings = F.normalize(aug_homo_user_embeddings[source_user] / self.tau)
        aug_homo_target_user_embeddings = F.normalize(aug_homo_user_embeddings[target_user] / self.tau)
        sim = torch.einsum('abc,adc->abd', homo_source_user_embeddings, aug_homo_source_user_embeddings).exp()
        loss_con += - torch.log(sim.diagonal(dim1=1, dim2=2) / sim.sum(-1)).mean()
        sim = torch.einsum('abc,adc->abd', homo_target_user_embeddings, aug_homo_target_user_embeddings).exp()
        loss_con += - torch.log(sim.diagonal(dim1=1, dim2=2) / sim.sum(-1)).mean()

        homo_source_item_embeddings = F.normalize(homo_item_embeddings[source_pos])
        homo_target_item_embeddings = F.normalize(homo_item_embeddings[target_pos])
        aug_homo_source_item_embeddings = F.normalize(aug_homo_item_embeddings[source_pos])
        aug_homo_target_item_embeddings = F.normalize(aug_homo_item_embeddings[target_pos])
        loss_con += InfoNCE(homo_source_item_embeddings, aug_homo_source_item_embeddings, self.tau)
        loss_con += InfoNCE(homo_target_item_embeddings, aug_homo_target_item_embeddings, self.tau)
        
        losses.append(self.cl_rate * loss_con)

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
