import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def st_topk(logits, budget, device):
    soft_output = F.softmax(logits, dim=-1)
    _, indices = torch.topk(soft_output, budget, dim=-1)
    hard_output = torch.zeros_like(logits).to(device)
    hard_output.scatter_(-1, indices, 1.0)
    return hard_output - soft_output.detach() + soft_output

class MLP(nn.Module):
    def __init__(self, input_dim, hid1, hid2, output_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_dim, hid1)
        self.l2 = nn.Linear(hid1, hid2)
        self.l3 = nn.Linear(hid2, output_dim)
        # self.dropout = dropout

        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)
        nn.init.kaiming_normal_(self.l3.weight)

        self.fc1 = nn.Sequential(
            self.l1,
            nn.LeakyReLU(),
            self.l2,
            nn.LeakyReLU(),
            self.l3
        )

    def forward(self, x):
        output = self.fc1(x)
        return output



class AttrGeneration(nn.Module):
    def __init__(self, labels, feat_dim, weight1, weight2, discrete, device, tar_num, feat_max, feat_min):
        super(AttrGeneration, self).__init__()
        self.labels = labels
        self.label_dim = labels.max().item() + 1
        self.feat_dim = feat_dim
        self.weight1 = weight1
        self.weight2 = weight2
        self.discrete = discrete
        self.device = device
        self.tar_num = tar_num
        self.feat_max = feat_max
        self.feat_min = feat_min
        # direct 方式
        self.obtain_feat = MLP(3 * self.label_dim + 2 * self.feat_dim, 128, 512, self.feat_dim)

    def pool_func(self, wlabel, wsec):
        sub_graph_emb = self.node_emb[self.sub_graph_nodes].mean(0)
        tmp_emb = F.relu(torch.mm(self.feat[self.target], self.weight1))

        tarfeat_emb = torch.mm(tmp_emb, self.weight2)
        if self.tar_num == 1:
            graph_emb = torch.cat((sub_graph_emb.unsqueeze(0), self.node_emb[self.target], tarfeat_emb, wlabel, wsec),
                                  1)
        else:
            tar_emb = self.node_emb[self.target].mean(0).unsqueeze(0)
            tarfeat_emb = tarfeat_emb.mean(0).unsqueeze(0)
            graph_emb = torch.cat((sub_graph_emb.unsqueeze(0), tar_emb, tarfeat_emb, wlabel.mean(0).unsqueeze(0),
                                   wsec.mean(0).unsqueeze(0)), 1)
        return graph_emb

    def forward(self, target, feat, sub_graph_nodes, node_emb, wlabel, wsec, feat_num=None, eps=1, train_flag=False):
        self.target = target
        self.node_emb = node_emb
        self.sub_graph_nodes = sub_graph_nodes
        self.feat = feat
        self.graph_embed = self.pool_func(wlabel, wsec)
        self.add_feat = self.obtain_feat(self.graph_embed).squeeze(0)
        inj_feat = self.add_feat.sigmoid()
        inj_feat = (self.feat_max - self.feat_min) * inj_feat + self.feat_min
        new_feat = torch.cat((self.feat, inj_feat.unsqueeze(0)), 0)
        return new_feat, inj_feat


class EdgeGeneration(nn.Module):
    def __init__(self, labels, feat_dim, weight1, weight2, device, tar_num=1):
        super(EdgeGeneration, self).__init__()
        self.labels = labels
        self.label_dim = self.labels.max() + 1
        self.feat_dim = feat_dim
        self.weight1 = weight1
        self.weight2 = weight2
        self.tar_num = tar_num
        # self.obtain_score = MLP(5*self.feat_dim+1, 1)
        self.obtain_score = MLP(3 * self.label_dim + 2 * self.feat_dim + tar_num, 512, 32, 1)
        self.device = device

    def concat(self, new_feat, wlabel, wsec):
        sub_xw = torch.mm(torch.mm(new_feat[self.sub_graph_nodes], self.weight1), self.weight2)
        tar_xw = torch.mm(torch.mm(new_feat[self.target], self.weight1), self.weight2)
        add_xw = torch.mm(torch.mm(new_feat[-1].unsqueeze(0), self.weight1), self.weight2)

        add_xw_rep = add_xw.repeat(len(self.sub_graph_nodes), 1)

        if self.adj_tensor.is_sparse:
            tar_norm_adj = self.adj_tensor[self.target.item()].to_dense()
            norm_a_target = tar_norm_adj[self.sub_graph_nodes].unsqueeze(1)
        elif self.adj_tensor.shape[1] == 1:
            norm_a_target = self.adj_tensor
        else:
            norm_a_target = self.adj_tensor[self.sub_graph_nodes, self.target].unsqueeze(0).t()

        tar_xw_rep = tar_xw.repeat(len(self.sub_graph_nodes), 1)
        w_rep = wlabel.repeat(len(self.sub_graph_nodes), 1)
        w_sec_rep = wsec.repeat(len(self.sub_graph_nodes), 1)

        concat_output = torch.cat((tar_xw_rep, sub_xw, add_xw_rep, norm_a_target, w_rep, w_sec_rep), 1)
        # concat_output = torch.cat((tar_emb_rep, sub_node_emb, add_emb_rep,tar_add_emb_sub), 1)
        return concat_output

    def forward(self, budget, target, sub_graph_nodes, new_feat, adj_tensor, wlabel, wsec, eps=0, train_flag=False):
        self.budget = budget
        self.adj_tensor = adj_tensor
        self.sub_graph_nodes = sub_graph_nodes
        self.target = target
        self.sub_cat_addnode = self.concat(new_feat, wlabel, wsec)
        self.output = self.obtain_score(self.sub_cat_addnode).transpose(0, 1)
        if self.output.dim() > 1:
            self.output = self.output.squeeze(0)
        elif self.output.dim() == 0:
            self.output = self.output.unsqueeze(0)

        if train_flag:
            score = st_topk(self.output.squeeze(), budget, self.device)
        else:
            _, indices = torch.topk(self.output.squeeze(), budget)
            score = torch.zeros_like(self.output.squeeze()).to(self.device)
            score[indices] = 1.0
        score_idx = torch.LongTensor(sub_graph_nodes.reshape(-1)).unsqueeze(0)

        return score, score_idx


class RATGNN(nn.Module):
    def __init__(self, labels, feat_dim, weight1, weight2, discrete, device, tar_num=1, feat_max=None, feat_min=None,
                 feat_num=None):
        super(RATGNN, self).__init__()
        self.labels = labels
        #         self.budget = budget
        self.feat_dim = feat_dim
        self.feat_num = feat_num
        self.add_node_agent = AttrGeneration(self.labels, self.feat_dim, weight1, weight2, discrete, device,
                                             tar_num, feat_max, feat_min).to(device)
        self.add_edge_agent = EdgeGeneration(self.labels, feat_dim, weight1, weight2, device, tar_num)
        self.tar_num = tar_num
        self.discrete = discrete
        self.device = device

    def add_node_and_update(self, feat_num, wlabel, wsec, eps=0, train_flag=False):

        return self.add_node_agent(self.target, self.feat, self.sub_graph_nodes, self.node_emb, wlabel, wsec, feat_num,
                                   eps, train_flag)

    def add_edge_and_update(self, new_feat, wlabel, wsec, eps=0, train_flag=False):

        return self.add_edge_agent(self.budget, self.target, self.sub_graph_nodes, new_feat, self.nor_adj_tensor,
                                   wlabel, wsec, eps, train_flag)

    def forward(self, target, sub_graph_nodes, budget, feat, nor_adj_tensor, node_emb, wlabel, wsec, train_flag, eps=0):
        self.target = target
        self.nor_adj_tensor = nor_adj_tensor
        self.sub_graph_nodes = sub_graph_nodes
        self.budget = budget
        self.feat = feat
        self.n = self.feat.shape[0]
        self.node_emb = node_emb
        if self.tar_num == 1:
            wlabel = wlabel.unsqueeze(0)
            wsec = wsec.unsqueeze(0)

        self.new_feat, self.add_feat = self.add_node_and_update(self.feat_num, wlabel, wsec, eps, train_flag=train_flag)

        self.score, self.masked_score_idx = self.add_edge_and_update(self.new_feat, wlabel, wsec, eps=eps,
                                                                     train_flag=train_flag)

        # Evaluation
        if train_flag:
            self.disc_score = self.score
        else:
            if self.discrete:
                feat_values, feat_indices = self.add_feat.topk(self.feat_num)
                self.disc_feat = torch.zeros_like(self.add_feat).to(self.device)
                self.disc_feat[feat_indices] = 1.
                self.new_feat[-1] = self.disc_feat

            edge_values, edge_indices = self.score.topk(budget)
            self.disc_score = torch.zeros_like(self.score).to(self.device)
            self.disc_score[edge_indices] = 1.

        return self.add_feat, self.disc_score, self.masked_score_idx

