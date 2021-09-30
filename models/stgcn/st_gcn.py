import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pdb
import time
import gc
from .gcn import GCN, GraphConvolution
from .st_graph import get_st_graph, get_spatial_graph
from .gat import GraphAttentionLayer, SpGraphAttentionLayer

def get_temporal_graph(rois_features_t0, rois_features_t1):
    sim_features = torch.matmul(rois_features_t0, rois_features_t1.transpose(0, 1))
    rois_t0_norms = rois_features_t0.norm(dim=1).unsqueeze(1)
    rois_t1_norms = rois_features_t1.norm(dim=1).unsqueeze(0)
    norms = torch.matmul(rois_t0_norms, rois_t1_norms)
    sim_features = sim_features / norms
    # sim_features = torch.exp(sim_features)
    sim_graph = F.softmax(sim_features, dim=-1)
    # print(sim_features)
    return sim_graph


class TemporalGAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, alpha=0.5, nheads=1, hidden_dim=1):
        """Dense version of GAT."""
        super(TemporalGAT, self).__init__()
        # self.compressor = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.attentions = [GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        print(len(self.attentions))
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_dim * nheads, output_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.compressor(x)
        print("----------------------")
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print("======================")
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj))
        return x

    
class TemporalGCN(torch.nn.Module):
    def __init__(self, in_channel=512, out_channel=512,
                 dropout=0.5, shortcut=False):
        super(TemporalGCN, self).__init__()

        # 1 by 1 conv -> 512  wang: 2048 -> 512
        self.out_channel = out_channel
        in_channel = in_channel  # 512
        dropout = dropout
        self.shortcut = shortcut
        # wang2018video differentiates forward graph and backward graph,
        # but in this implementation we ignore this.

        self.sim_gc1 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.sim_gc2 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.sim_gc3 = GraphConvolution(in_channel, self.out_channel, bias=False, batch=True)

        self.dropout = nn.Dropout(dropout)
        self.init_weight()

    def forward(self, input, adj):
        out = F.relu(self.sim_gc1(input, adj))
        #        out = self.dropout(out)
        if self.shortcut:
            out += input
        out = F.relu(self.sim_gc2(out, adj))

        if self.shortcut:
            out += input
        #        out = self.dropout(out)
        out = F.relu(self.sim_gc3(out, adj))
        return out

    def init_weight(self):
        nn.init.normal_(self.sim_gc1.weight.data, 0, 0.001)
        nn.init.normal_(self.sim_gc2.weight.data, 0, 0.001)
        nn.init.normal_(self.sim_gc3.weight.data, 0, 0.001)


class RGCN(torch.nn.Module):
    def __init__(self, in_channel=512, out_channel=512,
                 dropout=0.5, graph_model="gcn", stgcn_shortcut=False
                 ):
        super(RGCN, self).__init__()

        # 1 by 1 conv -> 512  wang: 2048 -> 512
        self.out_channel = out_channel
        in_channel = in_channel # 512


        # wang2018video differentiates forward graph and backward graph,
        # but in this implementation we ignore this.

        self.graph_model_type = graph_model
        if graph_model == "gcn":
            self.graph_model = TemporalGCN(in_channel, out_channel, dropout, shortcut=stgcn_shortcut)
        elif graph_model == "gat":
            self.graph_model = TemporalGAT(in_channel, out_channel, dropout)
        else:
            raise Exception("In Valid Graph Model")

        self.dropout = nn.Dropout(dropout)
        # self.init_weight()


    def generate_st_graphs(self, rois, connection, return_dict, st=0):
        for i, (r, c) in enumerate(zip(rois, connection)):
            return_dict[i+st] = get_st_graph(r,c)

    def forward(self, rois, rois_features, temporal_start_end=None):
        n_rois = rois.size(0)
        graph = torch.zeros(n_rois, n_rois)
        for i, (start_end_current, start_end_next) in enumerate(zip(temporal_start_end[:-1], temporal_start_end[1:])):
            rois_current = rois[start_end_current[0]:start_end_current[1]]
            rois_features_current = rois_features[start_end_current[0]: start_end_current[1]]
            rois_features_next = rois_features[start_end_next[0]: start_end_next[1]]
            spatial_graph = get_spatial_graph(rois_current)
            temporal_graph = get_temporal_graph(rois_features_current, rois_features_next)
            graph[start_end_current[0]:start_end_current[1], start_end_current[0]:start_end_current[1]] = spatial_graph
            graph[start_end_current[0]: start_end_current[1], start_end_next[0]: start_end_next[1]] = temporal_graph
        graph = graph.cuda()
        # graph = graph.detach()
        gc.collect()
        torch.cuda.empty_cache()
        sim_gcn = self.graph_model(rois_features, graph)
        temporal_features = []
        for start_end in temporal_start_end:
            temporal_nodes = sim_gcn[start_end[0]: start_end[1]]
            pooled_features = temporal_nodes.mean(0)
            if start_end[1] - start_end[0] == 0:
                pooled_features = torch.zeros_like(pooled_features)
            temporal_features.append(pooled_features)
        temporal_features = torch.stack(temporal_features)
        return temporal_features

    def sim_graph(self, features):
        sim1 = self.sim_embed1(features)
        sim2 = self.sim_embed2(features)
        sim_features = torch.matmul(sim1, sim2.transpose(1,2)) # d x d mat.
        sim_graph = F.softmax(sim_features, dim=-1)
        return sim_graph


    def get_optim_policies(self):
        normal_weight = []
        normal_bias = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif 'Conv' in str(type(m)):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]
