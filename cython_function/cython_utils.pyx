from __future__ import division
import numpy as np
import networkx as nx
cimport numpy as np
cimport cython
import scipy.sparse as ss
from skimage import measure
from skimage.segmentation import find_boundaries

ctypedef bint TYPE_BOOL
ctypedef unsigned long long TYPE_U_INT64
ctypedef int TYPE_INT32
ctypedef long TYPE_INT64
ctypedef float TYPE_FLOAT
ctypedef double TYPE_DOUBLE


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_max_iou(np.ndarray[TYPE_INT64, ndim=3] mask,
                 np.ndarray[TYPE_INT64, ndim=3] gt):
    cdef np.ndarray[TYPE_INT64, ndim=3] mask_iou
    cdef np.ndarray[TYPE_INT64, ndim=1] id_list
    cdef int id_max
    # cdef np.ndarray[TYPE_INT64, ndim=1] list

    mask_iou = mask*gt
    id_list = np.unique(mask_iou)
    if id_list[0]==0:
        id_list = id_list[1:]

    iou_id_list = [np.sum(np.where(mask_iou==i,1,0)) for i in id_list]
    if len(iou_id_list) ==0:
        return 0
    id_max_index = np.argmax(np.array(iou_id_list))
    id_max = id_list[id_max_index]
    return id_max

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def target2label(np.ndarray[TYPE_INT64, ndim=3] target,
                 np.ndarray[TYPE_INT64, ndim=3] segments,
                 np.ndarray[TYPE_INT64, ndim=2] adj):

    cdef np.ndarray[TYPE_INT64, ndim=2] label
    label = np.zeros_like(adj)
    cdef np.ndarray[TYPE_INT64, ndim=2] label_weight
    label = np.zeros_like(adj)
    label_weight = np.zeros_like(adj)
    cdef int n_nodes
    n_nodes= adj.shape[0]
    cdef int i
    cdef int j
    cdef np.ndarray[TYPE_INT64, ndim=3] binary_id1
    cdef np.ndarray[TYPE_INT64, ndim=3] binary_id2
    cdef np.ndarray[TYPE_INT64, ndim=3] binary_id
    cdef int ll
    cdef int max_iou_id1
    cdef int max_iou_id2
    cdef TYPE_INT64 w

    id_map = []
    weight_map = []
    for i in range(n_nodes):
        binary_id = np.where(segments == i, 1, 0)
        max_iou_id = find_max_iou(binary_id, target)
        id_map.append(max_iou_id)
        w0 = np.sum(binary_id)
        weight_map.append(w0)


    for i in range(n_nodes):
        for j in range(i+1):

            ll=0
            if adj[i,j] ==0 or j==0 or i==0 or i==j:
            #if adj[i,j] ==0 or i==j:  #no background
                ll=250
            else:
                max_iou_id1 = id_map[i]
                max_iou_id2 = id_map[j]
                ll = 1 if max_iou_id1 == max_iou_id2 and max_iou_id1!=0 else 0
            label[i,j] = ll
            label[j,i] = ll


    for i in range(n_nodes):
        for j in range(i + 1):
            if not (adj[i, j] == 0 or j == 0 or i == 0 or i == j) :

                w1 = weight_map[i]
                w2 = weight_map[j]
                w = (w1+w2)/2
                label_weight[i, j] = w
                label_weight[j, i] = w

    return label,label_weight

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_adj(np.ndarray[TYPE_INT64, ndim=2] segments,
            np.ndarray[TYPE_INT64, ndim=2] adj,
            boundary):
    cdef np.ndarray[long, ndim=2] adj_intensity
    adj_intensity = np.zeros_like(adj)

    cdef np.ndarray[TYPE_INT64, ndim=2] adj_num
    adj_num = np.zeros_like(adj)

    cdef int n_nodes
    n_nodes = adj.shape[0]
    cdef int i
    cdef int j
    cdef np.ndarray[TYPE_INT64, ndim=2] boundary1
    cdef np.ndarray[TYPE_INT64, ndim=2] boundary2
    cdef np.ndarray[TYPE_INT64, ndim=2] boundary_tmp
    cdef np.ndarray[TYPE_INT64, ndim=2] boundary_pair

    boundary_map = []
    for i in range(n_nodes):
        segments_tmp = segments == i
        boundary_tmp = find_boundaries(segments_tmp, mode='thick').astype(np.int64)

        boundary_map.append(boundary_tmp)

    boundary = boundary * 255
    for i in range(n_nodes):
        for j in range(i + 1):
            if adj[i, j] == 1:
                boundary1 = boundary_map[i]
                boundary2 = boundary_map[j]
                boundary_pair = boundary1 * boundary2 * boundary.astype(np.int64)
                intensity = np.sum(boundary_pair) / np.sum(boundary1 * boundary2)
                num = np.sum(boundary_pair > 0)
                adj_intensity[i, j] = intensity
                adj_intensity[j, i] = intensity
                adj_num[i, j] = num
                adj_num[j, i] = num

    return adj_num, adj_intensity