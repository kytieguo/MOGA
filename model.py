import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
# from GCNConv import GCNConv
# from SGConv import SGConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv, SGConv
from utils import *

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu1 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index.long())
        x = self.prelu1(x)
        return x

class DRP(nn.Module):
    def __init__(self, gcn_layer, device, omics, layersnums = 2, hidden_size=128, att_heads=2, dropout=0.2, emb_output=128, out_size=32,
                 omics_n=6, feat_dim_li = [256,256,256,256,256], units_list=[256, 256, 256], use_relu=True, use_bn=True, use_GMP=True):
        super(DRP, self).__init__()
        self.feature_dim = emb_output
        self.omics_n = omics_n
        self.omics = omics
        self.gat_in = emb_output * 2
        self.device = device
        self.units_list = units_list
        self.layer_nums = layersnums
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP

        self.conv1 = SGConv(gcn_layer, units_list[0])
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        self.encoder = Encoder(in_channels=128, hidden_channels=64)
        self.encoder_g = nn.ModuleList()
        self.predict = Classifier(out_size)
        self.test_layer = nn.Linear(hidden_size, out_size)
        self.encoder_g.append(
            GATConv(self.gat_in, hidden_size, attn_drop=dropout, residual=True, feat_drop=dropout,
                    num_heads=att_heads, activation=F.elu, allow_zero_in_degree=True))
        for i in range(1, self.layer_nums-1):
            in_dim = hidden_size * att_heads
            self.encoder_g.append(
                GATConv(in_dim, hidden_size, attn_drop=dropout, residual=True, feat_drop=dropout,
                        num_heads=att_heads, activation=F.elu, allow_zero_in_degree=True))

        self.encoder_g.append(
            GATConv(hidden_size * att_heads, hidden_size, attn_drop=dropout, residual=True, feat_drop=dropout,
                    num_heads=att_heads, activation=F.elu, allow_zero_in_degree=True))

        self.graph_conv = []
        self.graph_bn = []
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGConv(units_list[i], units_list[i + 1]).to(device))
            self.graph_bn.append(nn.BatchNorm1d((units_list[i + 1])).to(device))
        self.conv_end = SGConv(units_list[-1], emb_output)
        self.batch_end = nn.BatchNorm1d(emb_output)
        # --------cell line layers (three omics)
        # -------exp_layer
        # self.fc_gexp1 = nn.Linear(feat_dim_li[0], 256)
        # self.batch_gexp1 = nn.BatchNorm1d(256)
        # self.fc_gexp2 = nn.Linear(256, emb_output)
        self.exp_cov1 = nn.Conv2d(1, 128, (1, 50), stride=(1, 8))
        self.exp_cov2 = nn.Conv2d(128, 256, (1, 20), stride=(1, 5))
        self.fla_exp = nn.Flatten()
        self.fc_exp = nn.Linear(256, emb_output)
        # -------copy number_layer
        # self.fc_cn1 = nn.Linear(feat_dim_li[1], 256)
        # self.batch_cn1 = nn.BatchNorm1d(256)
        # self.fc_cn2 = nn.Linear(256, emb_output)
        self.meta_cov1 = nn.Conv2d(1, 128, (1, 10), stride=(1, 5))
        self.meta_cov2 = nn.Conv2d(128, 256, (1, 5), stride=(1, 3))
        self.fla_cn = nn.Flatten()
        self.fc_cn = nn.Linear(256, emb_output)
        # -------mut_layer
        self.cov1 = nn.Conv2d(1, 128, (1, 50), stride=(1, 8))
        self.cov2 = nn.Conv2d(128, 256, (1, 20), stride=(1, 5))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(256, emb_output)  # The 30 here should be the same as the 30 of the cov2 output
        # --------meta_layer
        self.fc_meta1 = nn.Linear(feat_dim_li[2], 256)
        self.batch_meta1 = nn.BatchNorm1d(256)
        self.fc_meta2 = nn.Linear(256, emb_output)
        # --------methy_layer
        self.fc_methy1 = nn.Linear(feat_dim_li[3], 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, emb_output)
        # --------prot_layer
        self.fc_port1 = nn.Linear(feat_dim_li[4], 256)
        self.batch_port1 = nn.BatchNorm1d(256)
        self.fc_port2 = nn.Linear(256, emb_output)
        # ------Concatenate_six omics
        self.fcat = nn.Linear(self.feature_dim * self.omics_n, emb_output)
        self.batchc = nn.BatchNorm1d(self.feature_dim)
        self.weight = nn.Parameter(torch.Tensor(256, 256))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(self.feature_dim, 64)  # The 64 here is outputchannal
        self.fd = nn.Linear(self.feature_dim, 64)  # The 64 here is outputchannal

        self.L = nn.Linear(256, 128)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        # reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def DPP_toplogy_Graph_Generation(self, data):
        data_copy = data.cpu()
        (A_indices, B_indices) = data_copy.reshape(2, -1)
        num_c_nodes = len(data)
        G_new_adjacency_matrix = np.zeros((num_c_nodes, num_c_nodes), dtype=int)
        for i in range(num_c_nodes):
            G_new_adjacency_matrix[i] = np.logical_or(A_indices == A_indices[i], B_indices == B_indices[i])
        edge = torch.tensor(G_new_adjacency_matrix.astype(int).nonzero()).permute(1, 0)
        return edge
    def random_agument(self, x, edge, aug_prob=0.1):
        drop_mask = torch.empty((x.size(1),),
                                dtype=torch.float32,
                                device=x.device).uniform_(0, 1) < aug_prob
        x = x.clone()
        x[:, drop_mask] = 0
        mask_rates = torch.FloatTensor(np.ones(len(edge)) * aug_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        edge = edge[mask_idx].permute(1, 0)
        _graph = dgl.graph([]).to(x.device)
        _graph.add_nodes(x.shape[0])
        _graph.add_edges(edge[0], edge[1])
        return x.to(self.device), _graph.to(self.device)

    def get_emb(self):
        for l in range(self.layer_nums-1):
            self._old_emb = self.encoder_g[l](self._old_graph, self._old_emb).flatten(1)
        self._old_emb = self.encoder_g[-1](self._old_graph, self._old_emb).mean(1)
        return self._old_emb

    def get_emb_test(self, x):
        self._old_emb = self.L(x.to(self.device))
        return self._old_emb

    def caculated_loss(self, h1, h2):

        z1 = ((h1 - h1.mean(0)) / (h1.std(0)))
        z2 = ((h2 - h2.mean(0)) / (h2.std(0)))

        std_x = h1.var(dim=0)
        std_y = h2.var(dim=0)
        std_loss = torch.sum(torch.sqrt((0.5 - std_x) ** 2)) / \
                   2 + torch.sum(torch.sqrt((0.5 - std_y) ** 2)) / 2
        N = z1.shape[0]
        c = torch.mm(z1, z2.T)
        c = c / (N ** 2)
        loss_inv = torch.diagonal(c).sum()
        return loss_inv + std_loss * 0.001



    def forward(self, data, drug_feature, drug_adj, batch, cell_feature, edge, index):
        data_copy = data.cpu()
        mutation_data = cell_feature.dataset.tensors[0].to(self.device)
        expr_data = cell_feature.dataset.tensors[1].to(self.device)
        copy_number_data = cell_feature.dataset.tensors[2].to(self.device)
        metabolomic_data = cell_feature.dataset.tensors[3].to(self.device)
        methylation_data = cell_feature.dataset.tensors[4].to(self.device)
        prot_data = cell_feature.dataset.tensors[5].to(self.device)
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_conv1(x_drug)
        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)

        if self.use_GMP:
            x_drug = global_max_pool(x_drug, batch)
        else:
            x_drug = global_mean_pool(x_drug, batch)

        # -----cell line representation
        # -----mutation representation
        x_mutation = torch.tanh(self.cov1(mutation_data))
        x_mutation = F.max_pool2d(x_mutation, (1, 10))

        x_mutation = F.relu(self.cov2(x_mutation))
        x_mutation = F.max_pool2d(x_mutation, (1, 20))
        x_mutation = self.fla_mut(x_mutation)
        x_mutation = F.relu(self.fc_mut(x_mutation))

        # ----expr representation
        # x_expr = torch.sigmoid(self.fc_gexp1(expr_data))
        # x_expr = self.batch_gexp1(x_expr)
        # x_expr = F.relu(self.fc_gexp2(x_expr))

        x_expr = torch.tanh(self.cov1(expr_data))
        x_expr = F.max_pool2d(x_expr, (1, 10))

        x_expr = F.relu(self.cov2(x_expr))
        x_expr = F.max_pool2d(x_expr, (1, 20))
        x_expr = self.fla_mut(x_expr)
        x_expr = F.relu(self.fc_mut(x_expr))

        # ----copy number representation
        # x_copy_number = torch.tanh(self.fc_cn1(copy_number_data))
        # x_copy_number = self.batch_cn1(x_copy_number)
        # x_copy_number = F.relu(self.fc_cn2(x_copy_number))
        x_copy_number = torch.tanh(self.cov1(copy_number_data))
        x_copy_number = F.max_pool2d(x_copy_number, (1, 10))

        x_copy_number = F.relu(self.cov2(x_copy_number))
        x_copy_number = F.max_pool2d(x_copy_number, (1, 20))
        x_copy_number = self.fla_mut(x_copy_number)
        x_copy_number = F.relu(self.fc_mut(x_copy_number))

        # ----meta representation
        # x_meta = torch.tanh(self.fc_meta1(metabolomic_data))
        # x_meta = self.batch_meta1(x_meta)
        # x_meta = F.relu(self.fc_meta2(x_meta))
        x_meta = torch.tanh(self.meta_cov1(metabolomic_data))
        x_meta = F.max_pool2d(x_meta, (1, 5))

        x_meta = F.relu(self.meta_cov2(x_meta))
        x_meta = F.max_pool2d(x_meta, (1, 2))
        x_meta = self.fla_mut(x_meta)
        x_meta = F.relu(self.fc_mut(x_meta))

        # ----meth representation
        # x_methy = torch.tanh(self.fc_methy1(methylation_data))
        # x_methy = self.batch_methy1(x_methy)
        # x_methy = F.relu(self.fc_methy2(x_methy))
        x_methy = torch.tanh(self.cov1(methylation_data))
        x_methy = F.max_pool2d(x_methy, (1, 10))

        x_methy = F.relu(self.cov2(x_methy))
        x_methy = F.max_pool2d(x_methy, (1, 20))
        x_methy = self.fla_mut(x_methy)
        x_methy = F.relu(self.fc_mut(x_methy))

        # ----port representation
        # x_prot = torch.tanh(self.fc_port1(prot_data))
        # x_prot = self.batch_port1(x_prot)
        # x_prot = F.relu(self.fc_port2(x_prot))
        x_prot = torch.tanh(self.cov1(prot_data))
        x_prot = F.max_pool2d(x_prot, (1, 10))

        x_prot = F.relu(self.cov2(x_prot))
        x_prot = F.max_pool2d(x_prot, (1, 20))
        x_prot = self.fla_mut(x_prot)
        x_prot = F.relu(self.fc_mut(x_prot))


        if self.omics == 'exp,mut,cn,meta,meth':
            # x_cell = x_expr
            x_cell = torch.cat((x_expr, x_mutation, x_copy_number, x_meta, x_methy), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        elif self.omics == 'exp,mut,cn,meta,prot':
            # x_cell = x_mutation
            x_cell = torch.cat((x_expr, x_mutation, x_copy_number, x_meta, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        elif self.omics == 'exp,mut,cn,meth,prot':
            # x_cell = x_copy_number
            x_cell = torch.cat((x_mutation, x_expr, x_copy_number, x_methy, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        elif self.omics == 'exp,mut,meta,meth,prot':
            # x_cell = x_meta
            x_cell = torch.cat((x_mutation, x_expr, x_meta, x_methy, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        elif self.omics == 'exp,cn,meta,meth,prot':
            # x_cell = x_methy
            x_cell = torch.cat((x_expr, x_copy_number, x_meta, x_methy, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        elif self.omics == 'mut,cn,meta,meth,prot':
            # x_cell = x_prot
            x_cell = torch.cat((x_mutation, x_copy_number, x_meta, x_methy, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))
        else:
            print('all omics')
            x_cell = torch.cat((x_mutation, x_expr, x_copy_number, x_meta, x_methy, x_prot), 1)
            x_cell = F.leaky_relu(self.fcat(x_cell))

        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batchc(x_all)

        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T).to(self.device)
        pos_z = self.encoder(x_all, pos_edge)
        cell_pos = pos_z[:index, ]
        drug_pos = pos_z[index:, ]
        cell_fea = self.fc(x_all[:index, ])
        drug_fea = self.fd(x_all[index:, ])
        cell_fea = torch.sigmoid(cell_fea)
        drug_fea = torch.sigmoid(drug_fea)
        cell_emb = torch.cat((cell_pos, cell_fea), 1).cpu()
        drug_emb = torch.cat((drug_pos, drug_fea), 1).cpu()
        # pos_adj = torch.matmul(cell_pos, drug_pos.t())
        # pos_adj = self.act(pos_adj)
        DPP_emb = torch.cat((drug_emb[data_copy[:, 1]], cell_emb[data_copy[:, 0]]), dim=-1)
        edge_g = self.DPP_toplogy_Graph_Generation(data_copy[:, :2]).to(DPP_emb.device)
        prob = 0.1
        graph1_feat, graph1 = self.random_agument(DPP_emb, edge_g, prob)
        graph2_feat, graph2 = self.random_agument(DPP_emb, edge_g, prob)
        edge_g = edge_g.permute(1, 0)
        self._old_graph = dgl.graph(data=(edge_g[0], edge_g[1])).to(self.device)
        self._old_emb = DPP_emb.to(self.device)
        for l in range(self.layer_nums-1):
            graph1_feat = self.encoder_g[l](graph1, graph1_feat).flatten(1)
            graph2_feat = self.encoder_g[l](graph2, graph2_feat).flatten(1)
        graph1_feat = self.encoder_g[-1](graph1, graph1_feat).mean(1)
        graph2_feat = self.encoder_g[-1](graph2, graph2_feat).mean(1)
        loss = self.caculated_loss(graph1_feat, graph2_feat)
        emb = self.get_emb()
        # emb = self.get_emb_test(DPP_emb)
        out = self.predict(emb, 1)

        return out, loss, emb

class Classifier(nn.Module):
    def __init__(self, nfeat):
        super(Classifier, self).__init__()
        self.L1 = nn.Linear(nfeat, nfeat * 2)
        self.L2 =  nn.Linear(nfeat * 2, 2)
        # self.L1 = nn.Linear(256, 128)
        # self.L2 = nn.Linear(128, 2)

    def forward(self, x,e = 0):
        out = nn.ELU()(self.L1(x))
        out = self.L2(out)
        return nn.Softmax(dim = 1)(out)
