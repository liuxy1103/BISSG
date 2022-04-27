import os
import tifffile
import numpy as np
from PIL import Image

def gen_affs(map1, map2=None, dir=0, shift=1, padding=True, background=False):
    if dir == 0 and map2 is None:
        raise AttributeError('map2 is none')
    map1 = map1.astype(np.float32)
    h, w = map1.shape
    if dir == 0:
        map2 = map2.astype(np.float32)
    elif dir == 1:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[shift:, :] = map1[:h-shift, :]
    elif dir == 2:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[:, shift:] = map1[:, :w-shift]
    else:
        raise AttributeError('dir must be 0, 1 or 2')
    dif = map2 - map1
    out = dif.copy()
    out[dif == 0] = 1
    out[dif != 0] = 0
    if background:
        out[map1 == 0] = 0
        out[map2 == 0] = 0
    if padding:
        if dir == 1:
            out[0, :] = (map1[0, :] > 0).astype(np.float32)
        if dir == 2:
            out[:, 0] = (map1[:, 0] > 0).astype(np.float32)
    return out

def im2col(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4): 
    # "we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing 
    # more than one positive segment ID (zero is reserved for background) is marked as background."
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis = 1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg

def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)

    idxkeep = (i**2+j**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood 

def seg_to_aff(seg, nhood=mknhood2d(1), pad='replicate'):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.float32)
    
    if len(shape) == 3: # 3D affinity
        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )
    elif len(shape) == 2: # 2D affinity
        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )

    if nEdge==3 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)
        aff[2,:,:,0] = (seg[:,:,0]>0).astype(aff.dtype)
    elif nEdge==2 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)

    return aff


if __name__ == '__main__':

    data_path = '/braindat/lab/liuxy/superpixel/A1'
    mode = 'train'
    dir = os.path.join(data_path, mode)
    id_num = os.listdir(dir)
    id_label = [f for f in id_num if 'label' in f]

    if not mode =='validation':
        id_label.sort(key=lambda x: int(x[5:8]))
    else:
        id_label.sort(key=lambda x: int(x[21:23]))

    for fn_l in id_label:
        img = Image.open(os.path.join(dir, fn_l))

        method = 'ours' # ours or official
        img[img == 0] = 65534  #把id=0的不看成是背景 或者像我这样 直接变成其他的id
        # dilate boundary or erode instance
        img = seg_widen_border(img, tsz_h=1)
        # obtain affinity
        if method == 'ours':
            affs_y = gen_affs(img, None, dir=1, shift=1, padding=True, background=True)
            affs_x = gen_affs(img, None, dir=2, shift=1, padding=True, background=True)
            affs_yx = np.stack([affs_y, affs_x], axis=0)
        elif method == 'official':
            affs_yx = seg_to_aff(img)
        else:
            raise NotImplementedError

        # obtain boundary
        boundary = 0.5 * (affs_yx[0] + affs_yx[1])
        boundary[boundary <= 0.5] = 0
        boundary = (boundary * 255).astype(np.uint8)
        Image.fromarray(boundary).save(os.path.join(dir, fn_l.replace('label','boundary')))