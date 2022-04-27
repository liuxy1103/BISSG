import torch
from torch import nn
import torch.nn.functional as F
from .unet2d_residual import ResidualUNet2D
from utils.utils_rag_matrix import *
from .GCNN_model import *
import numpy as np
from train_util import *
import time
sys.path.append('../third_party/cython')
from connectivity import enforce_connectivity
from utils.fragment import watershed, randomlabel
from utils.water import *
import cython_utils
from postprocessing import merge_small_object, merge_func, remove_samll_object
from data.data_segmentation import relabel
from skimage import io

class unet_egnn(nn.Module):
    def __init__(self, n_channels,nfeatures,graph_layers,args,n_emb=16,n_pos=9):
        super(unet_egnn, self).__init__()
        self.in_channels = n_channels
        self.n_features = nfeatures
        self.graph_layers = graph_layers
        self.args = args
        self.n_emb = n_emb
        self.n_pos = n_pos
        self.unet = ResidualUNet2D(self.in_channels,self.n_features)
        for p in self.parameters():
            p.requires_grad = False
        self.egnn = GraphNetwork(16+32+2,32+2,32+2,self.graph_layers,dropout=0.5)

    def forward(self, x,mode='validation',fg=None,minsize=10):
        out_boundary, out_emb, out_binary_seg,spix4  = self.unet(x) #(B,2,H,W)
        boundary = torch.sigmoid(out_boundary.detach())
        boundary = boundary[0]
        boundary = 1.0 - 0.5 * (boundary[0] + boundary[1])
        boundary = boundary.cpu().numpy()
        # segments = gen_fragment(boundary, radius=5)

        pred_mask_b = fg.squeeze().cpu().numpy()
        pred_mask_b = pred_mask_b.astype(np.uint8)

        mask = pred_mask_b
        segments = gen_fragment(boundary, radius=2)
        segments = segments * mask
        minsize = minsize
        segments = remove_small_ids(segments, minsize=minsize)
        segments = enforce_connectivity(segments[np.newaxis, ...].astype(np.int64), minsize, 10000000)[0]
        segments = segments * mask
        segments = remove_small_ids(segments, minsize=minsize)

        # io.imsave('mask.tif',mask)
        # io.imsave('spixel.tif', segments.astype(np.uint16))
        # pred_mask_b = remove_samll_object(pred_mask_b)
        # segments = (segments) * mask
        inverse1, pack1,counts = np.unique(segments, return_inverse=True,return_counts=True)
        #print(len(inverse1))
        if len(inverse1) <= 1:
            return out_boundary,out_emb,out_binary_seg,None,None,segments,None


        # print('IDs:',len(np.unique(segments)))
        inverse1, pack1= np.unique(segments, return_inverse=True)
        segments = segments[np.newaxis,...]
        segments = segments[np.newaxis, ...]

        pack1 = pack1.reshape(segments.shape)
        inverse1 = np.arange(0, len(inverse1))
        segments = inverse1[pack1]

        time0 = time.time()

        node_feat, adj = Segments2RAG(segments[:,0,:],torch.cat((out_emb,spix4),1)) #boundary.shape(448,448)
        adj_num,adj_intensity = cython_utils.get_adj(segments[0,0],adj.cpu().numpy().astype(np.int64),boundary)
        adj_num = adj_num/adj_num.max()
        adj_intensity = adj_intensity/255

        adj_num = torch.tensor(adj_num).cuda().unsqueeze(0)
        adj_intensity = torch.tensor(adj_intensity).cuda().unsqueeze(0)
        adj_boundary = torch.cat((adj_intensity,adj_num)).float() #(2,n,n)
        time1 = time.time()
        print('Segments2RAG Time:',time1-time0)

        # adj_tmp = adj.clone()
        # adj[0, :]=0
        # adj[:, 0] = 0
        # adj = ((torch.mm(adj, adj)) > 0).float()
        # adj[0, :] = adj_tmp[0, :]
        # adj[:, 0] = adj_tmp[:, 0]

        adj_b = adj * (1 - adj_boundary[0])
        edge_feat = adj_b + 0
        edge_feat_neg = adj * (adj_boundary[0])
        edge_feat_all = torch.cat((edge_feat_neg.unsqueeze(0),edge_feat.unsqueeze(0)))

        edge_feat_list, node_feat_list = self.egnn(node_feat.unsqueeze(0),edge_feat_all.unsqueeze(0),adj,adj_boundary.unsqueeze(0))

        edge_feat_list = torch.cat(edge_feat_list)
        node_feat_list = torch.cat(node_feat_list)

        return out_boundary,out_emb,out_binary_seg,edge_feat_list,node_feat_list,segments,adj,spix4

if __name__ == '__main__':
    import numpy as np

    x = torch.Tensor(np.random.random((1, 1, 256, 256)).astype(np.float32)).cuda()

    # x = torch.Tensor(np.random.random((1, 1, 100, 256, 256)).astype(np.float32)).cuda()
    # mask = torch.Tensor(np.random.random((1, 1, 100, 256, 256)).astype(np.float32)).cuda()

    # x = torch.Tensor(np.random.random((1, 1, 66, 320, 320)).astype(np.float32)).cuda()
    # mask = torch.Tensor(np.random.random((1, 1, 66, 320, 320)).astype(np.float32)).cuda()

    # model = RSUNet_Nested([16,32,48,64,80]).cuda()
    # model = ResidualUNet2D(1,[3, 16, 32, 64,128])
    model = eval('ResidualUNet2D(1,[16,32,64,128,256])').cuda()

    # with torch.no_grad():
    out_o, out_c,out_s = model(x)
    print(out_o.shape, out_c.shape,out_s.shape)
