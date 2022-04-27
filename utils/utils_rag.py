# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

from pathlib import Path
import SimpleITK as sitk
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def log_args(args,log):
    args_info = "\n##############\n"
    for key in args.__dict__:
        args_info = args_info+(key+":").ljust(25)+str(args.__dict__[key])+"\n"
    args_info += "##############"
    log.info(args_info)

def save_nii(img,save_name):
    nii_image = sitk.GetImageFromArray(img)
    name = str(save_name).split("/")
    sitk.WriteImage(nii_image,str(save_name))
    print(name[-1]+" saving finished!")


def get_graph_from_image(segments,embedding): #Tensor
    """
    segments: HxW
    embedding: FxHxW

    #猝： 很多麻烦的东西，构建这样一个图只是用torch
    """

    NP_TORCH_FLOAT_DTYPE = torch.float32
    NP_TORCH_LONG_DTYPE = torch.int64
    NUM_FEATURES = embedding.shape[0]+len(segments.shape)
    # load the segments and convert it to rag
    num_nodes = torch.max(segments)
    nodes = {
        node: {
            "emb_list": [],
            "pos_list": []
        } for node in range(num_nodes + 1)
    }

    height = segments.shape[0]
    width = segments.shape[1]
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            emb = embedding[:,y, x]
            pos = torch.tensor([float(x) / width, float(y) / height]).cuda()
            nodes[node.item()]["emb_list"].append(emb)
            nodes[node.item()]["pos_list"].append(pos)
        # end for
    # end for

    G = nx.Graph()

    for node in nodes:
        nodes[node]["emb_list"] = torch.stack(nodes[node]["emb_list"])
        nodes[node]["pos_list"] = torch.stack(nodes[node]["pos_list"])
        # emb
        emb_mean = torch.mean(nodes[node]["emb_list"], dim=0)
        # rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        # rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = torch.mean(nodes[node]["pos_list"], dim=0)
        # pos_std = np.std(nodes[node]["pos_list"], axis=0)
        # pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug

        features = torch.cat(
            [
                torch.reshape(emb_mean, (-1,)),
                # np.reshape(rgb_std, -1),
                # np.reshape(rgb_gram, -1),
                torch.reshape(pos_mean, (-1,)),
                # np.reshape(pos_std, -1),
                # np.reshape(pos_gram, -1)
            ]
        )
        G.add_node(node, features=list(features))
    # end

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    # segments_ids = np.unique(segments)
    #
    # # centers
    # centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    vs_right = torch.cat([segments[:, :-1].flatten().unsqueeze(0), segments[:, 1:].flatten().unsqueeze(0)],dim=0)
    vs_below = torch.cat([segments[:-1, :].flatten().unsqueeze(0), segments[1:, :].flatten().unsqueeze(0)],dim=0)

    bneighbors = torch.unique(torch.cat([vs_right, vs_below]), dim=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i].item(), bneighbors[1, i].item())

    # Self loops
    for node in nodes:
        G.add_edge(node, node)

    n = len(G.nodes)
    m = len(G.edges)
    h = torch.zeros([n, NUM_FEATURES]).type(NP_TORCH_FLOAT_DTYPE)
    edges = torch.zeros([2 * m, 2]).type(NP_TORCH_LONG_DTYPE)
    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t

        edges[m + e, 0] = t
        edges[m + e, 1] = s
    # end for
    for i in G.nodes:
        h[i, :] = torch.tensor(G.nodes[i]["features"])
    # end for
    del G
    return h, edges #节点和边

def batch_graphs(gs): #input batch-size graphs from function:get_graph_from_image
    """
    Assure that every different graph have no identical IDS.
    """
    NP_TORCH_FLOAT_DTYPE = torch.float32
    NP_TORCH_LONG_DTYPE = torch.int64
    NUM_FEATURES = gs[0][0].shape[-1] #a node
    G = len(gs) #batch_size
    N = sum(g[0].shape[0] for g in gs) #number of all nodes from different graph
    M = sum(g[1].shape[0] for g in gs) #number of all relations-edges from different graph
    adj = torch.zeros([N, N]) # big adjacent matrix
    src = torch.zeros([M])
    tgt = torch.zeros([M])
    Msrc = torch.zeros([N, M])
    Mtgt = torch.zeros([N, M])
    Mgraph = torch.zeros([N, G])
    h = torch.cat([g[0] for g in gs]) #all nodes

    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0] #number of  nodes from one graph
        m = g[1].shape[0] #number of  edegs from one graph

        for e, (s, t) in enumerate(g[1]): #edges from one graph
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1

            src[m_acc + e] = n_acc + s #node1
            tgt[m_acc + e] = n_acc + t #node2

            Msrc[n_acc + s, m_acc + e] = 1 #node1-nodes(including many repeated nodes)
            Mtgt[n_acc + t, m_acc + e] = 1 #node2-nodes

        Mgraph[n_acc:n_acc + n, g_idx] = 1

        n_acc += n
        m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )



def Segments2RAG(segments,embedding):
    "segments: BxHxW"
    "embedding: BxCxHxW"
    NP_TORCH_FLOAT_DTYPE = np.float32
    NP_TORCH_LONG_DTYPE = np.int64
    NUM_FEATURES = embedding.shape[1] + +len(segments.shape)-1
    Batch_size = embedding.shape[0]

    #convert to new id without overlap between different Batch
    max_id_list = [segments[b].max() for b in range(Batch_size)]
    for b in range(Batch_size):
        if b>0:
            segments[b] = segments[b]+max_id_list[b-1]
            segments[b][segments[b]==max_id_list[b-1]]=0

    ## load one segments and convert it to one rag
    # for b in range(Batch_size):
    #     g = get_graph_from_image(segments[b],embedding[b])
    gs = [get_graph_from_image(segments[b],embedding[b])  for b in range(Batch_size)] #graph list
    G = len(gs)  # batch_size
    N = sum(g[0].shape[0] for g in gs) #number of all nodes from different graph
    adj = torch.zeros([N, N]).cuda() # big adjacent matrix
    h = torch.cat([g[0] for g in gs]).cuda() #all nodes
    n_acc = 0

    for g_idx, g in enumerate(gs):
        for e, (s, t) in enumerate(g[1]):  # edges from one graph
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1

    return h,adj

if __name__ == "__main__":
    import h5py
    import torch
    spixel_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\ID\0000.tif'
    segments = io.imread(spixel_path)
    segments = torch.tensor(segments.astype(np.int64))
    emb_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\embedding\0001.hdf'
    with h5py.File(emb_path,'r') as f:
        embedding = f['main'][:]
    embedding = torch.tensor(embedding)
    print(embedding.shape)
    # graph = get_graph_from_image(segments,embedding)
    graph = Segments2RAG(segments.unsqueeze(0),embedding.unsqueeze(0))
    #graph = Segments2RAG(torch.cat((segments.unsqueeze(0),segments.unsqueeze(0))), torch.cat((embedding.unsqueeze(0),embedding.unsqueeze(0))))
    print(graph[0].shape,graph[0].dtype,graph[1].shape,graph[1].dtype)

