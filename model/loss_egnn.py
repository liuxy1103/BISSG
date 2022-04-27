import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
import cython_utils
import higra as hg


# def find_max_iou(mask,gt):
#     mask_iou = mask * gt
#     id_list = list(np.unique(mask_iou))
#     id_list.remove(0)
#     iou_id_list = [np.sum(np.where(mask_iou == i, 1, 0)) for i in id_list]
#     if len(iou_id_list) == 0:
#         # print('no iou')
#         return 0
#     id_max_index = np.argmax(iou_id_list)
#     id_max = id_list[id_max_index]
#     return id_max
#
# def target2label(output,target,segments,adj):
#     segments = segments.squeeze(0)
#     # print(output.shape,target.shape,segments.shape,adj.shape)
#     target = np.array(target.cpu())
#     label = torch.zeros(output.shape[-2:])
#     n_nodes = output.shape[-1]
#     for i in range(n_nodes):
#         for j in range(i+1):
#             if adj[i,j]==0 or j==0 or i==0 or i==j:
#                 ll=250
#                 #print('skip')
#             else:
#                 binary_id1 = np.where(np.array(segments) == i, 1, 0)
#                 binary_id2 = np.where(np.array(segments) == j, 1, 0)
#                 max_iou_id1 = find_max_iou(binary_id1,target)
#                 max_iou_id2 = find_max_iou(binary_id2,target)
#                 ll = 1 if max_iou_id1 == max_iou_id2 and max_iou_id1!=0  else 0
#             # if ll!=250:
#                 # print(ll)
#             label[i,j] = ll
#             label[j,i] = ll
#
#     return label

def egnn_BCE(output, output_feat, target, segments, adj):
    """
    output: shape(n_layers,2,n_nodes,n_nodes)
    output_feat: shape(n_layers,N-D,n_nodes)
    """
    gnn_layers = output.shape[0]
    time0 = time.time()
    # label = target2label(output,target,segments,adj)
    label, label_weight = cython_utils.target2label(target.cpu().numpy().astype(np.int64),
                                                    segments[:, 0, :].astype(np.int64),
                                                    adj.cpu().numpy().astype(
                                                        np.int64))  # label : shape(n_nodes,n_nodes)
    time1 = time.time()
    print('Target2label Time:', time1 - time0)

    label_weight = torch.tensor(label_weight)
    label_weight = label_weight.repeat(gnn_layers, 1, 1).cuda().float()

    label = torch.tensor(label)
    label = label.repeat(gnn_layers, 1, 1).cuda()

    weight = torch.FloatTensor([torch.sum(label == 1).item(), torch.sum(label == 0).item()]).cuda()
    # weight = torch.FloatTensor(weight_rate)
    loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=250)
    loss = loss_fn(output, label.long())

    print('egnn label weight:', weight)
    if loss == 0:
        print('current egnn loss:', loss)

    output = F.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)  # (B,n,n)
    output_symmetric = output.transpose(1, 2)
    consistency_loss = torch.mean(torch.abs(output_symmetric - output).float())
    print('EGNN loss-edge:{}, consistency loss:{}'.format(loss, consistency_loss))
    # node constraint
    #     delta = 0.5
    delta_d = 4
    # import pdb
    # pdb.set_trace()

    x_i = output_feat.unsqueeze(2)  # n_layersxNx1xf
    x_j = torch.transpose(x_i, 1, 2)  # n_layersx1xNxf
    x_ij = x_i - x_j  # n_layersxNxNxf

    x_ij = x_ij.float()
    label_node1 = label == 1

    # L2 norm
    ori_shape = x_ij.shape
    x_ij = torch.reshape(x_ij, (-1, ori_shape[-1]))  # (n_layersxNxN)xf
    x_ij = torch.norm(x_ij, dim=1)

    x_ij = x_ij.reshape(ori_shape[:-1])  # n_layersxNxN

    x_ij1 = x_ij * label_node1.float()  # n_layersxNxNxf  #x_ij0.max()=3.847
    # x_ij1= F.relu(x_ij1 - delta)**2
    if not torch.sum(label == 1).item()==0:
        print('Disance1:', x_ij1[label_node1].min(), x_ij1.max())
        x_ij1 = x_ij1 ** 2
        x_ij1 = x_ij1 * label_node1.float()
        loss_egnn_node1 = torch.sum(x_ij1) / torch.sum(label_node1)
    else:
        loss_egnn_node1 = torch.sum(x_ij1*0)

    label_node0 = label == 0
    x_ij0 = x_ij * label_node0.float()  # x_ij0.max()=5
    if not torch.sum(label == 0).item() == 0:
        print('Disance0:', x_ij0[label_node0].min(), x_ij0.max())
        x_ij0 = F.relu(delta_d - x_ij0) ** 2
        x_ij0 = x_ij0 * label_node0.float()
        loss_egnn_node0 = torch.sum(x_ij0) / torch.sum(label_node0)
    else:
        loss_egnn_node0 = torch.sum(x_ij0 * 0)

    #
    loss_egnn_node = loss_egnn_node0 + loss_egnn_node1
    # loss_egnn_node = loss_egnn_node0
    print('EGNN loss-Node:{}, loss_egnn_node0 loss:{}, loss_egnn_node1 loss:{}'.format(loss_egnn_node, loss_egnn_node0,
                                                                                       loss_egnn_node1))
    # print('EGNN loss-Node:{}'.format(loss_egnn_node))

    return loss + consistency_loss + 0.1 * loss_egnn_node


def Graph_MALIS(output, output_feat, target, segments, adj):
    """
    output: shape(n_layers,2,n_nodes,n_nodes)
    adj: (n_nodes,n_nodes)
    """
    output = F.softmax(output, dim=1)

    output = output[:, 1]
    gnn_layers = output.shape[0]
    n_nodes = adj.shape[0]
    label, label_weight = cython_utils.target2label(target.cpu().numpy().astype(np.int64),
                                                    segments[:, 0, :].astype(np.int64),
                                                    adj.cpu().numpy().astype(
                                                        np.int64))  # label : shape(n_nodes,n_nodes)
    label = torch.tensor(label)
    weight = torch.FloatTensor([torch.sum(label==1).item(),torch.sum(label==0).item()]).cuda()

    print('egnn label weight:',weight)


    criterion = torch.nn.MSELoss()
    loss_malis = 0
    for l in range(gnn_layers):
        out = output[l]
        g = hg.UndirectedGraph(n_nodes)
        vertex1 = []
        vertex2 = []
        edge_weights = []
        all_pairs = []
        gt = []
        for i in range(n_nodes):
            for j in range(i + 1):
                if adj[i, j] == 1:
                    vertex1.append(i)
                    vertex2.append(j)
                    if i == 0 or j == 0:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(out[i, j].item())
                    if (not (i == 0 or j == 0 or i == j)) and adj[i, j] == 1:
                        all_pairs.append([i, j])
                        gt.append(label[i, j])
        #print('all_pairs:',all_pairs)
        if len(all_pairs)==0:
            gt = torch.tensor(gt).cuda()
            loss = torch.sum(gt.float() * 0)
        else:
            all_pairs = np.vstack(all_pairs)
            gt = torch.tensor(gt).cuda()
            vertex1 = tuple(vertex1)
            vertex2 = tuple(vertex2)
            g.add_edges(vertex1, vertex2)
            edge_weights = np.array(edge_weights)
            tree, altitudes = hg.bpt_canonical(g, -edge_weights)  # Input graph must be connected
            lcaf = hg.make_lca_fast(tree)
            # Get edge
            mst = hg.get_attribute(tree, "mst")
            mst_map = hg.get_attribute(mst, "mst_edge_map")
            mst_idcs = lcaf.lca(*all_pairs.T) - tree.num_leaves()
            edge_idcs = mst_map[mst_idcs]
            #print("edge_idcs",edge_idcs)

            if isinstance(edge_idcs,np.ndarray):
                mm_edges = np.array([g.edge_from_index(edge_idx)[:2] for edge_idx in edge_idcs])
                ims = np.amin(mm_edges, axis=1)
                jms = np.amax(mm_edges, axis=1)

                mm_values = torch.stack(
                    [out[im, jm] for im, jm in
                     zip(ims, jms)])
                loss = criterion(mm_values, gt.float())
            else:
                loss = torch.sum(gt.float() * 0)
        loss_malis = loss_malis + loss

    print('Graph-Malis loss:{}'.format(loss_malis))

    label = label.repeat(gnn_layers, 1, 1).cuda()
    # node constraint
    #     delta = 0.5
    delta_d = 3
    # import pdb
    # pdb.set_trace()

    x_i = output_feat.unsqueeze(2)  # n_layersxNx1xf
    x_j = torch.transpose(x_i, 1, 2)  # n_layersx1xNxf
    x_ij = x_i - x_j  # n_layersxNxNxf

    x_ij = x_ij.float()
    label_node1 = label == 1

    # L2 norm
    ori_shape = x_ij.shape
    x_ij = torch.reshape(x_ij, (-1, ori_shape[-1]))  # (n_layersxNxN)xf
    x_ij = torch.norm(x_ij, dim=1)

    x_ij = x_ij.reshape(ori_shape[:-1])  # n_layersxNxN

    x_ij1 = x_ij * label_node1.float()  # n_layersxNxNxf  #x_ij0.max()=3.847
    # x_ij1= F.relu(x_ij1 - delta)**2
    if not torch.sum(label == 1).item() == 0:
        print('Disance1:', x_ij1[label_node1].min(), x_ij1.max())
        x_ij1 = F.relu(x_ij1 - delta_d) ** 2
        x_ij1 = x_ij1 * label_node1.float()
        loss_egnn_node1 = torch.sum(x_ij1) / torch.sum(label_node1)
    else:
        loss_egnn_node1 = torch.sum(x_ij1 * 0)

    label_node0 = label == 0
    x_ij0 = x_ij * label_node0.float()  # x_ij0.max()=5
    if not torch.sum(label == 0).item() == 0:
        print('Disance0:', x_ij0[label_node0].min(), x_ij0.max())
        x_ij0 = F.relu(delta_d - x_ij0) ** 2
        x_ij0 = x_ij0 * label_node0.float()
        loss_egnn_node0 = torch.sum(x_ij0) / torch.sum(label_node0)
    else:
        loss_egnn_node0 = torch.sum(x_ij0 * 0)

    #
    loss_egnn_node = loss_egnn_node0 + loss_egnn_node1
    # loss_egnn_node = loss_egnn_node0
    print('EGNN loss-Node:{}, loss_egnn_node0 loss:{}, loss_egnn_node1 loss:{}'.format(loss_egnn_node, loss_egnn_node0,
                                                                                       loss_egnn_node1))
    # print('EGNN loss-Node:{}'.format(loss_egnn_node))

    return loss_malis + 0.1 * loss_egnn_node
