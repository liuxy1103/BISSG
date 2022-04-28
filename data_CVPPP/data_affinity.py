import numpy as np


# from Janelia pyGreentea
# https://github.com/naibaf7/PyGreentea

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
    tsz = 2 * tsz_h + 1
    sz = seg.shape
    if len(sz) == 3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            seg[z] = seg[z] * ((p0 == p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg, ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg


def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad + 1, 1)
    y = np.arange(-ceilrad, ceilrad + 1, 1)
    [i, j] = np.meshgrid(y, x)

    idxkeep = (i ** 2 + j ** 2) <= radius ** 2
    i = i[idxkeep].ravel();
    j = j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i) / 2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood


def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad + 1, 1)
    y = np.arange(-ceilrad, ceilrad + 1, 1)
    z = np.arange(-ceilrad, ceilrad + 1, 1)
    [i, j, k] = np.meshgrid(z, y, x)

    idxkeep = (i ** 2 + j ** 2 + k ** 2) <= radius ** 2
    i = i[idxkeep].ravel();
    j = j[idxkeep].ravel();
    k = k[idxkeep].ravel();
    zeroIdx = np.array(len(i) // 2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx], i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def mknhood3d_aniso(radiusxy=1, radiusxy_zminus1=1.8):
    # Makes nhood structures for some most used dense graphs.
    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
    nhood = np.zeros((nhoodxyz.shape[0] + 2 * nhoodxy_zminus1.shape[0], 3), dtype=np.int32)
    nhood[:3, :3] = nhoodxyz
    nhood[3:, 0] = -1
    nhood[3:, 1:] = np.vstack((nhoodxy_zminus1, -nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)


def seg_to_aff(seg, nhood=mknhood2d(1), pad='replicate'):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.float32)  # (e, z, y, x)

    if len(shape) == 3:  # 3D affinity
        '''
        (16, 160, 159)
        (16, 159, 160)
        (15, 160, 160)
        '''
        for e in range(nEdge):
            aff[e, \
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
                 max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
                 max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] == \
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
                 max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
                 max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]) \
                * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
                   max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
                   max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] > 0) \
                * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
                   max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
                   max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])] > 0)  # positive mask for affinity map
    elif len(shape) == 2:  # 2D affinity
        for e in range(nEdge):
            aff[e, \
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
                 max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] == \
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
                 max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1])]) \
                * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
                   max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] > 0) \
                * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
                   max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1])] > 0)

    if nEdge == 3 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)
        aff[2, :, :, 0] = (seg[:, :, 0] > 0).astype(aff.dtype)
    elif nEdge == 2 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)

    return aff
