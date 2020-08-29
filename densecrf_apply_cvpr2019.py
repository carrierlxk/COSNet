#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:37:37 2019

@author: xiankai
"""

import pydensecrf.densecrf as dcrf
import numpy as np
import sys
import os


from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from os import listdir, makedirs
from os.path import isfile, join
from multiprocessing import Process

            
def worker(scale, g_dim, g_factor,s_dim,C_dim,c_factor):
    davis_path = '/home/xiankai/work/DAVIS-2016/JPEGImages/480p'#'/home/ying/tracking/pdb_results/FBMS-results'
    origin_path = '/home/xiankai/work/DAVIS-2016/Results/Segmentations/480p/COS-78.2'#'/home/xiankai/work/DAVIS-2016/Results/Segmentations/480p/ECCV'#'/media/xiankai/Data/segmentation/match-Weaksup_VideoSeg/result/test/davis_iteration_conf_sal_match_scale/COS/'
    out_folder = '/home/xiankai/work/DAVIS-2016/Results/Segmentations/480p/cvpr2019_crfs'#'/media/xiankai/Data/ECCV-crf'#'/home/xiankai/work/DAVIS-2016/Results/Segmentations/480p/davis_ICCV_new/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    origin_file = listdir(origin_path)
    origin_file.sort()
    for i in range(0, len(origin_file)):
        d = origin_file[i]
        vidDir = join(davis_path, d)
        out_folder1 = join(out_folder,'f'+str(scale)+str(g_dim)+str(g_factor)+'_'+'s'+str(s_dim)+'_'+'c'+str(C_dim)+str(c_factor))
        resDir = join(out_folder1, d)
        if not os.path.exists(resDir):
                os.makedirs(resDir)
        rgb_file = listdir(vidDir)
        rgb_file.sort()
        for ii in range(0,len(rgb_file)):  
            f = rgb_file[ii]
            img = imread(join(vidDir, f))
            segDir = join(origin_path, d)
            frameName = str.split(f, '.')[0]
            anno_rgb = imread(segDir + '/' + frameName + '.png').astype(np.uint32)
            min_val = np.min(anno_rgb.ravel())
            max_val = np.max(anno_rgb.ravel())
            out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
            labels = np.zeros((2, img.shape[0], img.shape[1]))
            labels[1, :, :] = out
            labels[0, :, :] = 1 - out
    
            colors = [0, 255]
            colorize = np.empty((len(colors), 1), np.uint8)
            colorize[:,0] = colors
            n_labels = 2
    
            crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
    
            U = unary_from_softmax(labels,scale)
            crf.setUnaryEnergy(U)
    
            feats = create_pairwise_gaussian(sdims=(g_dim, g_dim), shape=img.shape[:2])
    
            crf.addPairwiseEnergy(feats, compat=g_factor,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    
            feats = create_pairwise_bilateral(sdims=(s_dim,s_dim), schan=(C_dim, C_dim, C_dim),# 30,5
                                          img=img, chdim=2)
    
            crf.addPairwiseEnergy(feats, compat=c_factor,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    
            #Q = crf.inference(5)
            Q, tmp1, tmp2 = crf.startInference()
            for i in range(5):
                #print("KL-divergence at {}: {}".format(i, crf.klDivergence(Q)))
                crf.stepInference(Q, tmp1, tmp2)
    
            MAP = np.argmax(Q, axis=0)
            MAP = colorize[MAP]
            
            imsave(resDir + '/' + frameName + '.png', MAP.reshape(anno_rgb.shape))
            print ("Saving: " + resDir + '/' + frameName + '.png')
scales = [1]#[0.5,1]#[0.1,0.3,0.5,0.6]#[0.5, 1.0]
g_dims = [1]#[1,3]#[1,3]
g_factors =[5]#[3,5,10] #[ 3, 5,10]
s_dims = [10,15,20] #[5,10,20]#[11, 12, 13]#[9,10,11] #10
Cs = [7]#[5]#[8]# [ 7,8,9,10] #8
b_factors = [8,9,10]
for scale in scales: 
    for g_dim in g_dims:
        for ii in range(0,len(g_factors)):
            g_factor = g_factors[ii]
            for jj in range(0,len(s_dims)):
                s_dim = s_dims[jj]
                for cs in Cs:
                    p1 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, b_factors[0]))
                    p2 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, b_factors[1]))
                    p3 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, b_factors[2]))
                    #p4 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, 4))
                    #p5 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, 1))
                    #p6 = Process(target = worker, args = (scale, g_dim, g_factor,s_dim,cs, 1))
                    
                    p1.start()
                    p2.start()
                    p3.start()
                    #p4.start()
                    #p5.start()
                    #p6.start()
            
            
