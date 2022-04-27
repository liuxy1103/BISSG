from collections import OrderedDict
import math
#import seaborn as sns
import torch.nn as nn
import torch
import torch.nn.functional as F


class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0) #batch_size?
        num_data = node_feat.size(1)  #num_samples

        # get eye matrix (batch_size x 2 x node_size x node_size)  adjacent matrix without self-loop
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda()

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1) #各个边上的权重归一化

        # compute attention and aggregate
        '''
        分开处理edge_feat的两种特征
        '''
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat) #batch级别的矩阵相乘

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features+2, #add two channels
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features+2,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat,adj_boundary):
        # compute abs(x_i, x_j)

        '''
        edge_feat: bx2xnxn
        node_feat:bxnxf
        '''
        x_i = node_feat.unsqueeze(2) # bxnx1xf
        x_j = torch.transpose(x_i, 1, 2) #bx1xnxf
        x_ij = torch.abs(x_i - x_j) #bxnxnxf
        x_ij = torch.transpose(x_ij, 1, 3)#bxfxnxn

        x_ij = torch.cat((x_ij,adj_boundary),dim=1) #[1, 36, n, n]

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        edge_feat = edge_feat + force_edge_feat  ##add EYE
        edge_feat = edge_feat + 1e-6  #防止出现0
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features, #32+3+2
                 num_layers,
                 dropout=0.0): #num_embding_features,num_edge_features,num_node_features,num_layers=3
        #num_embding_features -->l-->num_node_features
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout


        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat,adj,adj_boundary):
        # for each layer

        edge_feat_list = []
        node_feat_list = []
        for l in range(self.num_layers):# 三次循环进行特征融合
            # (1) edge to node
            LN_node = nn.LayerNorm(node_feat.size()[1:]).cuda()
            node_feat = LN_node(node_feat)
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat, adj_boundary)

            edge_feat = edge_feat * adj
            # save edge feature
            edge_feat_list.append(edge_feat)
            node_feat_list.append(node_feat)

        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))


        return edge_feat_list,node_feat_list

if __name__ == '__main__':
    import h5py
    from skimage import io
    from utils.utils_rag import *
    spixel_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\ID\0000.tif'
    segments = io.imread(spixel_path)
    emb_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\embedding\0001.hdf'
    with h5py.File(emb_path,'r') as f:
        embedding = f['main'][:]
    segments,embedding = segments[0:512, 0:512],embedding[:,0:512,0:512]
    inverse1, pack1 = np.unique(segments, return_inverse=True)
    pack1 = pack1.reshape(segments.shape)
    inverse1 = np.arange(0, inverse1.size)
    segments = inverse1[pack1]

    segments = torch.tensor(segments.astype(np.int64)).unsqueeze(0)
    embedding = torch.tensor(embedding).unsqueeze(0)

    node_feat, edge_feat = Segments2RAG(segments,embedding)
    print(node_feat.shape,edge_feat.shape)
    model = GraphNetwork(node_feat.shape[1],node_feat.shape[1],node_feat.shape[1],1).cuda()
    edge_feat_list = model(node_feat.unsqueeze(0).cuda(), edge_feat.repeat(2,1,1).unsqueeze(0).cuda())
    print(edge_feat_list)