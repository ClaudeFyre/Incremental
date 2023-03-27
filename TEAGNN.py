import torch
import torch.nn as nn
from torch import softmax
import torch.nn.functional as F
from torch_sparse import SparseTensor

class TEAGNNLayer(nn.Module):
    def __init__(self, node_size, rel_size, time_size, ent_size, depth=2, attn_heads=1, attn_heads_reduction='concat',
                 activation='relu', attn_kernel_initializer='glorot_uniform', attn_kernel_regularizer=None,
                 attn_kernel_constraint=None):
        super(TEAGNNLayer, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.time_size = time_size
        self.ent_size = ent_size
        self.depth = depth
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        self.attn_kernel_initializer = attn_kernel_initializer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.attn_kernel_constraint = attn_kernel_constraint

        self.attn_kernels = nn.ModuleList([])
        for l in range(self.depth):
            self.attn_kernels.append(nn.ModuleList([]))
            for head in range(self.attn_heads):
                attn_kernel = nn.Parameter(torch.Tensor(3 * self.ent_size, 1))
                attn_kernel_time = nn.Parameter(torch.Tensor(3 * self.ent_size, 1))
                nn.init.xavier_uniform_(attn_kernel)
                nn.init.xavier_uniform_(attn_kernel_time)
                self.attn_kernels[l].append([attn_kernel, attn_kernel_time])

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        time_emb = inputs[2]
        adj_indices = inputs[3]._indices()
        adj_values = inputs[3]._values()
        adj = SparseTensor(row=adj_indices[0], col=adj_indices[1], value=adj_values,
                           sparse_sizes=(self.node_size, self.node_size))
        sparse_indices = inputs[4]._indices()
        sparse_val = inputs[5]._values()
        t_sparse_indices = inputs[6]._indices()

        outputs.append(features)
        features = self.activation(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]

                rels_sum = SparseTensor(row=sparse_indices[0], col=sparse_indices[1], value=sparse_val,
                                        sparse_sizes=(self.triple_size, self.rel_size))
                rels_sum = torch.sparse.mm(rels_sum, rel_emb)
                neighs = features[adj_indices[1]]
                selfs = features[adj_indices[0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                bias = torch.sum(neighs * rels_sum, dim=1, keepdim=True) * rels_sum
                neighs = neighs - 2 * bias
                att = torch.squeeze(torch.mm(torch.cat([selfs, neighs, rels_sum], dim=-1), attention_kernel[0]), dim=-1)
                att_values = F.embedding(adj_indices[1], att)
                att = SparseTensor(row=adj_indices[0], col=adj_indices[1], value=att_values,
                                   sparse_sizes=(self.node_size, self.node_size))

                times_sum = SparseTensor(row=t_sparse_indices[0], col=t_sparse_indices[1], value=sparse_val,
                                        sparse_sizes=(self.time_size, self.time_size))
                times_sum = torch.sparse.mm(times_sum, time_emb)
                neighs_t = features[adj_indices[1]]
                selfs_t = features[adj_indices[0]]
                times_sum = F.normalize(times_sum, p=2, dim=1)
                bias_t = torch.sum(neighs_t * times_sum, dim=1, keepdim=True) * times_sum
                neighs_t = neighs_t - 2 * bias_t
                att_t = torch.squeeze(torch.mm(torch.cat([selfs_t, neighs_t, times_sum], dim=-1), attention_kernel[1]),
                                      dim=-1)
                att_values_t = F.embedding(adj_indices[1], att_t)
                att_t = SparseTensor(row=adj_indices[0], col=adj_indices[1], value=att_values_t,
                                     sparse_sizes=(self.node_size, self.node_size))

                att = att + att_t
                att = softmax(att)

                neighs = torch.sparse.mm(att, features)
                if self.use_bias:
                    biases = self.bias_kernels[l][head](neighs)
                    neighs = neighs + biases

                features_list.append(neighs)

            features = torch.stack(features_list, dim=1)
            features = torch.mean(features, dim=1)
            if self.use_layer_norm:
                features = self.layer_norms[l](features)

            features = self.activation(features)
            outputs.append(features)

        return outputs
