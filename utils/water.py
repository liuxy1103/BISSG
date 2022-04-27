import os
import cv2
import mahotas
import tifffile
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

def get_seeds(boundary, method='grid', next_id=1, radius=5, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        Bc = np.ones((radius,radius))
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance, Bc=Bc)
        seeds, num_seeds = mahotas.label(maxima, Bc=Bc)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds

def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the number of instances is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    color_pred = color_pred
    return color_pred

def gen_fragment(boundary, radius=5):
    seeds, _ = get_seeds(boundary, next_id=1, radius=radius, method='maxima_distance')
    fragments = mahotas.cwatershed(boundary, seeds)
    return fragments

def remove_small_ids(segments,minsize=20):
    inverse1, pack1, counts = np.unique(segments, return_inverse=True, return_counts=True)
    small_ids = inverse1[counts < minsize]
    up0 = segments.shape[0] - 1
    up1 = segments.shape[1] - 1
    for i in small_ids:
        X_co, Y_co = np.where(segments == i)
        new_id1 = segments[np.minimum(X_co.max() + 1, up0), np.minimum(Y_co.max(), up1)]
        new_id2 = segments[np.maximum(X_co.min(), 0), np.maximum(Y_co.min() - 1, 0)]
        if np.sum(segments == new_id1) > np.sum(segments == new_id2):
            segments[segments == i] = new_id1
        else:
            segments[segments == i] = new_id2

    return segments

# def gen_fragment2(boundary):
#     import elf.segmentation.watershed as ws
#     fragments, _ = ws.distance_transform_watershed(boundary, threshold=.25, sigma_seeds=2.)
#     return fragments
if __name__=='__main__':
    method = 1 # 1 or 2
    boundary = np.asarray(Image.open('./boundary.png'))
    # boundary = 1, background = 0
    boundary = 1 - boundary.astype(np.float32) / 255.0

    # obtain fragements
    if method == 1:
        fragments = gen_fragment(boundary, radius=5) # radius is used to control the number of generated fragments
    # else:
    #     fragments = gen_fragment2(boundary)
    # Mask background
    label = tifffile.imread('plant043_label.tif')
    mask = label != 0
    mask = binary_erosion(mask, iterations=1, border_value=True)
    fragments[mask==False] = 0
    # display
    color_frag = draw_fragments_2d(fragments)
    cv2.imwrite('./fragments.png', color_frag)
