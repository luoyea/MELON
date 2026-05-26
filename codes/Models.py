from distutils.command.build import build
from locale import normalize
from re import A
import numpy as np
from time import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import pandas as pd
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from utility.parser import parse_args 
from modules.modality_quality import ModalityQualityScorer, AspectDynamicGate, AspectFusion
args = parse_args()

def normalize_laplacian(edge_index, edge_weight):
    num_nodes = maybe_num_nodes(edge_index)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_weight

class Our_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Our_GCNs, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

from torch_geometric.nn.inits import uniform
class Nonlinear_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Nonlinear_GCNs, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, weight_vector, size=None):
        x = torch.matmul(x, self.weight)
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

class ICEnRCE(nn.Module):
    def __init__(self, n_users, n_items, n_layers, has_norm, feat_embed_dim, image_feats, text_feats, train_items, delta, adj):
        super(ICEnRCE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.delta = delta
        self.train_items = train_items
        self.n_trains = 0
        self.items = []
        self.rel_embed_dim = args.rel_embed_dim

        user_index = {}
        start_index = 0
        for uid, items in train_items.items():
            self.items.extend(items)
            end_index = start_index+len(items)
            temp_users = [i for i in range(start_index, end_index)]
            user_index[uid] = temp_users
            start_index = end_index

        user_view = []
        for uid, items in user_index.items():
            for item in items:
                temp_user = [0 for i in range(self.n_users)]
                temp_user[uid] = 1
                user_view.append(temp_user)

        item_index = {}
        for i in range(self.n_items):
            temp_i_index = list(filter(lambda x: self.items[x] == i, range(len(self.items))))
            item_index[i] = temp_i_index

        item_view = []
        for iid in self.items:
            temp_item = [0 for i in range(self.n_items)]
            temp_item[iid] = 1
            item_view.append(temp_item)

        user_view = torch.FloatTensor(user_view)
        item_view = torch.FloatTensor(item_view)

        self.image_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.text_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)

        self.image_query = nn.Embedding(self.n_users, self.rel_embed_dim)
        self.text_query = nn.Embedding(self.n_users, self.rel_embed_dim)
        nn.init.xavier_uniform_(self.image_query.weight)
        nn.init.xavier_uniform_(self.text_query.weight)

        self.image_embedding = nn.Embedding.from_pretrained(torch.tensor(image_feats, dtype=torch.float), freeze=True)
        self.item_image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.v_rel_mlp = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(image_feats.shape[1]*2, self.rel_embed_dim)))

        self.text_embedding = nn.Embedding.from_pretrained(torch.tensor(text_feats, dtype=torch.float), freeze=True)
        self.item_text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)
        self.t_rel_mlp = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(text_feats.shape[1]*2, self.rel_embed_dim)))

        self.uv_agg = torch.mm(adj.cpu().detach(), self.image_embedding.weight)
        self.ut_agg = torch.mm(adj.cpu().detach(), self.text_embedding.weight)

        u_inter_img = torch.mm(user_view, self.uv_agg)
        i_inter_img = torch.mm(item_view, self.image_embedding.weight)
        u_inter_txt = torch.mm(user_view, self.ut_agg)
        i_inter_txt = torch.mm(item_view, self.text_embedding.weight)    

        self.image_rel = torch.cat([u_inter_img, i_inter_img], dim=1)
        self.text_rel = torch.cat([u_inter_txt, i_inter_txt], dim=1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_rel = self.image_rel.to(device)
        self.text_rel = self.text_rel.to(device)

        self.uv_agg = self.uv_agg.to(device)
        self.ut_agg = self.ut_agg.to(device)

        self.layers = nn.ModuleList([Our_GCNs(self.feat_embed_dim, self.feat_embed_dim) for _ in range(self.n_layers)])

    def forward(self, edge_index, edge_weight, users, neg_items, eval=False):

        image_emb = self.item_image_trs(self.image_embedding.weight)  
        text_emb = self.item_text_trs(self.text_embedding.weight)  

        if self.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)

        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight

        # propagate
        ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
        ego_text_emb = torch.cat([text_preference, text_emb], dim=0)

        for layer in self.layers:
            side_image_emb = layer(ego_image_emb, edge_index, edge_weight)
            side_text_emb = layer(ego_text_emb, edge_index, edge_weight)

            ego_image_emb = side_image_emb + self.delta * ego_image_emb
            ego_text_emb = side_text_emb + self.delta * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)

        image_rel = self.image_rel
        comp_rel_v = torch.mm(image_rel, self.v_rel_mlp)

        text_rel = self.text_rel
        comp_rel_t = torch.mm(text_rel, self.t_rel_mlp)

        image_query = self.image_query.weight
        text_query = self.text_query.weight

        if len(neg_items) != 1: 
            image_neg_samples = torch.cat([self.uv_agg[users],self.image_embedding.weight[neg_items]], dim=1) 
            compressed_img_negsams = torch.mm(image_neg_samples, self.v_rel_mlp)

            text_neg_samples = torch.cat([self.ut_agg[users],self.text_embedding.weight[neg_items]], dim=1)
            compressed_txt_negsams = torch.mm(text_neg_samples, self.t_rel_mlp)
        else:  
            compressed_img_negsams = self.uv_agg
            compressed_txt_negsams = self.ut_agg

        if eval:
            return ego_image_emb, ego_text_emb

        items = torch.cat([final_image_emb, final_text_emb], dim=1)  
        user_preference = torch.cat([final_image_preference, final_text_preference], dim=1)  

        return user_preference, items, image_query, text_query, comp_rel_v, comp_rel_t, compressed_img_negsams, compressed_txt_negsams, self.v_rel_mlp, self.t_rel_mlp, self.image_embedding.weight, self.text_embedding.weight
    
class MCE(nn.Module):
    def __init__(self, n_users, n_items, n_layers, has_norm, feat_embed_dim, delta):
        super(MCE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.delta = delta
        self.image_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.text_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)

        self.image_repre = nn.Embedding(self.n_items, self.feat_embed_dim)
        self.text_repre = nn.Embedding(self.n_items, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_repre.weight)
        nn.init.xavier_uniform_(self.text_repre.weight)

        self.layers = nn.ModuleList([Our_GCNs(self.feat_embed_dim, self.feat_embed_dim) for _ in range(self.n_layers)])

    def forward(self, edge_index_img, edge_weight_img, edge_index_txt, edge_weight_txt, eval=False):

        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight

        image_repre = self.image_repre.weight
        text_repre = self.text_repre.weight

        # propagate
        ego_image_emb = torch.cat([image_preference, image_repre], dim=0)
        ego_text_emb = torch.cat([text_preference, text_repre], dim=0)

        for layer in self.layers: #self.layers from nn.ModuleList
            side_image_emb = layer(ego_image_emb, edge_index_img, edge_weight_img)
            side_text_emb = layer(ego_text_emb, edge_index_txt, edge_weight_txt)

            ego_image_emb = side_image_emb + self.delta * ego_image_emb
            ego_text_emb = side_text_emb + self.delta * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)
   
        if eval:
            return ego_image_emb, ego_text_emb

        items = torch.cat([final_image_emb, final_text_emb], dim=1) 
        user_preference = torch.cat([final_image_preference, final_text_preference], dim=1)  
        
        return user_preference, items

class MELON(nn.Module):
    def __init__(self, n_users, n_items, feat_embed_dim, nonzero_idx, nonzero_idx_img, nonzero_idx_txt, has_norm, image_feats, text_feats, train_items, n_layers, alpha, beta, gamma, delta,
                 use_mqm=True, use_aqg=True, fixed_weight=False, n_aspects=4, eta_mq=0.2, lambda_ent=1e-3, mq_hidden_dim=128, q_lambda=1.0):
        super(MELON, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.feat_embed_dim = feat_embed_dim
        self.n_layers = n_layers
        self.nonzero_idx = nonzero_idx
        self.nonzero_idx_img = nonzero_idx_img
        self.nonzero_idx_txt = nonzero_idx_txt
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_mqm = bool(use_mqm)
        self.use_aqg = bool(use_aqg)
        self.fixed_weight = bool(fixed_weight)
        self.n_aspects = int(n_aspects)
        self.eta_mq = float(eta_mq)
        self.lambda_ent = float(lambda_ent)
        self.mq_hidden_dim = int(mq_hidden_dim)
        self.q_lambda = float(q_lambda)
        self.enable_mq_component = self.eta_mq > 0 and (self.use_mqm or self.use_aqg or self.fixed_weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.feat_embed_dim % self.n_aspects != 0:
            raise ValueError('feat_embed_dim ({}) must be divisible by n_aspects ({}).'.format(self.feat_embed_dim, self.n_aspects))

        nonzero_idx = torch.tensor(self.nonzero_idx, device=self.device).long().T
        nonzero_idx[1] = nonzero_idx[1] + self.n_users
        nonzero_idx_img = torch.tensor(self.nonzero_idx_img, device=self.device).long().T
        nonzero_idx_img[1] = nonzero_idx_img[1] + self.n_users
        nonzero_idx_txt = torch.tensor(self.nonzero_idx_txt, device=self.device).long().T
        nonzero_idx_txt[1] = nonzero_idx_txt[1] + self.n_users        

        self.edge_index = torch.cat([nonzero_idx, torch.stack([nonzero_idx[1], nonzero_idx[0]], dim=0)], dim=1)
        self.edge_weight = torch.ones((self.edge_index.size(1)), device=self.device).view(-1, 1)
        self.edge_weight = normalize_laplacian(self.edge_index, self.edge_weight)

        self.edge_index_img = torch.cat([nonzero_idx_img, torch.stack([nonzero_idx_img[1], nonzero_idx_img[0]], dim=0)], dim=1)
        self.edge_weight_img = torch.ones((self.edge_index_img.size(1)), device=self.device).view(-1, 1)
        self.edge_weight_img = normalize_laplacian(self.edge_index_img, self.edge_weight_img)

        self.edge_index_txt = torch.cat([nonzero_idx_txt, torch.stack([nonzero_idx_txt[1], nonzero_idx_txt[0]], dim=0)], dim=1)
        self.edge_weight_txt = torch.ones((self.edge_index_txt.size(1)), device=self.device).view(-1, 1)
        self.edge_weight_txt = normalize_laplacian(self.edge_index_txt, self.edge_weight_txt)

        nonzero_idx = torch.tensor(self.nonzero_idx, device=self.device).long().T
        self.adj = torch.sparse.FloatTensor(
            nonzero_idx,
            torch.ones((nonzero_idx.size(1)), device=self.device),
            (self.n_users, self.n_items)
        ).to_dense().to(self.device)

        self.icerce = ICEnRCE(self.n_users, self.n_items, self.n_layers, has_norm, self.feat_embed_dim, image_feats, text_feats, train_items, self.delta, self.adj) 
        self.mce = MCE(self.n_users, self.n_items, self.n_layers, has_norm, self.feat_embed_dim, self.delta)

        # Modality quality-aware light-weight extension.
        self.img_quality_scorer = ModalityQualityScorer(self.feat_embed_dim, image_feats.shape[1], self.mq_hidden_dim)
        self.txt_quality_scorer = ModalityQualityScorer(self.feat_embed_dim, text_feats.shape[1], self.mq_hidden_dim)
        self.aspect_gate = AspectDynamicGate(quality_scale=self.q_lambda)
        self.aspect_fusion = AspectFusion()
        self.reset_gate_monitor()

    def forward(self, users, neg_items, eval=False): 
        if eval:                  
            user_ice, item_ice, image_query, text_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, v_rel_mlp, t_rel_mlp, image_feats, text_feats = self.icerce(self.edge_index, self.edge_weight, users, neg_items, eval=True)            
            user_mce, item_mce = self.mce(self.edge_index_img, self.edge_weight_img, self.edge_index_txt, self.edge_weight_txt, eval=True)
          
            return user_ice, item_ice, user_mce, item_mce, image_query, text_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, v_rel_mlp, t_rel_mlp, image_feats, text_feats                                                          

        user_ice, item_ice, image_query, text_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, v_rel_mlp, t_rel_mlp, image_feats, text_feats = self.icerce(self.edge_index, self.edge_weight, users, neg_items, eval=False)                           
        user_mce, item_mce = self.mce(self.edge_index_img, self.edge_weight_img, self.edge_index_txt, self.edge_weight_txt, eval=False)

        return user_ice, item_ice, user_mce, item_mce, image_query, text_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, v_rel_mlp, t_rel_mlp, image_feats, text_feats          

    def reset_gate_monitor(self):
        self.gate_monitor = {
            'sum_w_img': 0.0,
            'sum_w_txt': 0.0,
            'sum_q_img': 0.0,
            'sum_q_txt': 0.0,
            'count': 0
        }

    def get_gate_monitor(self):
        if self.gate_monitor['count'] == 0:
            return {'w_img': 0.0, 'w_txt': 0.0, 'q_img': 0.0, 'q_txt': 0.0}
        c = float(self.gate_monitor['count'])
        stats = {
            'w_img': self.gate_monitor['sum_w_img'] / c,
            'w_txt': self.gate_monitor['sum_w_txt'] / c,
            'q_img': self.gate_monitor['sum_q_img'] / c,
            'q_txt': self.gate_monitor['sum_q_txt'] / c
        }
        self.reset_gate_monitor()
        return stats

    def _split_aspect(self, emb):
        # emb: [B, D] -> [B, A, D/A]
        aspect_dim = emb.size(1) // self.n_aspects
        return emb.view(emb.size(0), self.n_aspects, aspect_dim)

    def _to_index_tensor(self, item_ids, device):
        if torch.is_tensor(item_ids):
            return item_ids.long().to(device)
        return torch.tensor(item_ids, dtype=torch.long, device=device)

    def _compute_quality_and_gate(self, user_emb, item_emb, item_ids):
        """
        user_emb/item_emb: [B, 2*feat_embed_dim]
        item_ids: [B]
        """
        device = user_emb.device
        item_ids = self._to_index_tensor(item_ids, device)

        u_img, u_txt = torch.split(user_emb, self.feat_embed_dim, dim=1)
        i_img, i_txt = torch.split(item_emb, self.feat_embed_dim, dim=1)
        raw_img = self.icerce.image_embedding.weight[item_ids]
        raw_txt = self.icerce.text_embedding.weight[item_ids]

        if self.use_mqm:
            q_img = self.img_quality_scorer(u_img, i_img, raw_img)
            q_txt = self.txt_quality_scorer(u_txt, i_txt, raw_txt)
        else:
            q_img = torch.full((u_img.size(0),), 0.5, device=device, dtype=u_img.dtype)
            q_txt = torch.full((u_txt.size(0),), 0.5, device=device, dtype=u_txt.dtype)

        u_img_a = self._split_aspect(u_img)
        u_txt_a = self._split_aspect(u_txt)
        i_img_a = self._split_aspect(i_img)
        i_txt_a = self._split_aspect(i_txt)

        w_img, w_txt = self.aspect_gate(
            u_img_a, i_img_a, u_txt_a, i_txt_a, q_img, q_txt,
            use_aqg=self.use_aqg,
            fixed_weight=self.fixed_weight
        )
        s_mq = self.aspect_fusion(u_img_a, i_img_a, u_txt_a, i_txt_a, w_img, w_txt)
        return s_mq, w_img, w_txt, q_img, q_txt

    def _gate_entropy_reg(self, w_img, w_txt):
        if self.fixed_weight:
            return torch.zeros(1, device=w_img.device, dtype=w_img.dtype).squeeze(0)
        weights = torch.stack([w_img, w_txt], dim=2)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=2)
        max_entropy = float(np.log(2.0))
        return max_entropy - torch.mean(entropy)

    def _update_gate_monitor(self, w_img, w_txt, q_img, q_txt):
        self.gate_monitor['sum_w_img'] += float(w_img.mean().detach().cpu())
        self.gate_monitor['sum_w_txt'] += float(w_txt.mean().detach().cpu())
        self.gate_monitor['sum_q_img'] += float(q_img.mean().detach().cpu())
        self.gate_monitor['sum_q_txt'] += float(q_txt.mean().detach().cpu())
        self.gate_monitor['count'] += 1

    def compute_mq_score_for_eval(self, batch_user_emb, batch_item_emb, item_batch):
        """
        Used in evaluation:
          batch_user_emb: [U, 2D]
          batch_item_emb: [I, 2D]
          item_batch: iterable with I indices
        Return:
          [U, I] modality quality-aware scores
        """
        n_users = batch_user_emb.size(0)
        n_items = batch_item_emb.size(0)
        user_expand = batch_user_emb.repeat_interleave(n_items, dim=0)
        item_expand = batch_item_emb.repeat(n_users, 1)
        item_ids = self._to_index_tensor(list(item_batch), batch_item_emb.device).repeat(n_users)
        s_mq, _, _, _, _ = self._compute_quality_and_gate(user_expand, item_expand, item_ids)
        return s_mq.view(n_users, n_items)

    def bpr_loss(self, user_ice, item_ice, user_mce, item_mce, image_query, text_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, users, pos_items, neg_items, pos_pairs):
        current_ice_user_emb = user_ice[users]
        ice_pos_item_emb = item_ice[pos_items]
        ice_neg_item_emb = item_ice[neg_items]

        current_mce_user_emb = user_mce[users]
        mce_pos_item_emb = item_mce[pos_items]
        mce_neg_item_emb = item_mce[neg_items]        

        img_user_standpoint = image_query[users]
        txt_user_standpoint = text_query[users] 
        current_cmp_rel_v = comp_rel_v[pos_pairs]
        current_cmp_rel_t = comp_rel_t[pos_pairs]
        user_standpoint = torch.cat([img_user_standpoint, txt_user_standpoint], dim=1)
        current_cmp_rel = torch.cat([current_cmp_rel_v, current_cmp_rel_t], dim=1)
        current_cmp_rel_neg = torch.cat([comp_rel_v_neg, comp_rel_t_neg], dim=1)

        # target-aware
        item_item = torch.mm(item_ice, item_ice.T)
        pos_item_query = item_item[pos_items, :] # (batch_size, n_items)
        neg_item_query = item_item[neg_items, :] # (batch_size, n_items)
        pos_target_user_alpha = torch.softmax(torch.multiply(pos_item_query, self.adj[users, :]).masked_fill(self.adj[users, :] == 0, -1e9), dim=1) # (batch_size, n_items)
        neg_target_user_alpha = torch.softmax(torch.multiply(neg_item_query, self.adj[users, :]).masked_fill(self.adj[users, :] == 0, -1e9), dim=1) # (batch_size, n_items)
        pos_target_user = torch.mm(pos_target_user_alpha, item_ice) # (batch_size, dim) 
        neg_target_user = torch.mm(neg_target_user_alpha, item_ice) # (batch_size, dim) 

        # predictor
        pos_rce_score = torch.sum(torch.mul(user_standpoint, current_cmp_rel), dim=1)
        neg_rce_score = torch.sum(torch.mul(user_standpoint, current_cmp_rel_neg), dim=1) 
        base_pos_scores = (1 - self.gamma) * torch.sum(torch.mul(current_ice_user_emb, ice_pos_item_emb), dim=1) + self.gamma * torch.sum(torch.mul(pos_target_user, ice_pos_item_emb), dim=1) + self.alpha * torch.sum(torch.mul(current_mce_user_emb, mce_pos_item_emb), dim=1)+self.beta*pos_rce_score
        base_neg_scores = (1 - self.gamma) * torch.sum(torch.mul(current_ice_user_emb, ice_neg_item_emb), dim=1) + self.gamma * torch.sum(torch.mul(neg_target_user, ice_neg_item_emb), dim=1) + self.alpha * torch.sum(torch.mul(current_mce_user_emb, mce_neg_item_emb), dim=1)+self.beta*neg_rce_score

        reg_loss = torch.zeros(1, device=base_pos_scores.device, dtype=base_pos_scores.dtype).squeeze(0)
        if self.enable_mq_component:
            s_mq_pos, w_img_pos, w_txt_pos, q_img_pos, q_txt_pos = self._compute_quality_and_gate(current_ice_user_emb, ice_pos_item_emb, pos_items)
            s_mq_neg, w_img_neg, w_txt_neg, q_img_neg, q_txt_neg = self._compute_quality_and_gate(current_ice_user_emb, ice_neg_item_emb, neg_items)
            pos_scores = base_pos_scores + self.eta_mq * s_mq_pos
            neg_scores = base_neg_scores + self.eta_mq * s_mq_neg
            gate_reg_pos = self._gate_entropy_reg(w_img_pos, w_txt_pos)
            gate_reg_neg = self._gate_entropy_reg(w_img_neg, w_txt_neg)
            reg_loss = self.lambda_ent * 0.5 * (gate_reg_pos + gate_reg_neg)
            self._update_gate_monitor(w_img_pos, w_txt_pos, q_img_pos, q_txt_pos)
            self._update_gate_monitor(w_img_neg, w_txt_neg, q_img_neg, q_txt_neg)
        else:
            pos_scores = base_pos_scores
            neg_scores = base_neg_scores

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        ice_regularizer = 1./2*(ice_pos_item_emb**2).sum() + 1./2*(ice_neg_item_emb**2).sum() + 1./2*(current_ice_user_emb**2).sum()
        mce_regularizer = 1./2*(mce_pos_item_emb**2).sum() + 1./2*(mce_neg_item_emb**2).sum() + 1./2*(current_mce_user_emb**2).sum()
        rce_regularizer = 1./2*(user_standpoint**2).sum() + 1./2*(current_cmp_rel**2).sum() + 1./2*(current_cmp_rel_neg**2).sum()  
        ice_emb_loss = ice_regularizer / ice_pos_item_emb.size(0)
        mce_emb_loss = mce_regularizer / mce_pos_item_emb.size(0)
        rce_emb_loss = rce_regularizer / current_cmp_rel.size(0) 
        emb_loss = ice_emb_loss+mce_emb_loss+rce_emb_loss

        return mf_loss, emb_loss, reg_loss
