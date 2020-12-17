# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:53:20 2018

@author: carri
"""

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from dataloaders import PairwiseImg_video_test_try as db
#from dataloaders import StaticImg as db #采用voc dataset的数据设置格式方法
import matplotlib.pyplot as plt
import random
import timeit
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from deeplab.siamese_model_conf_try import CoattentionNet
from torchvision.utils import save_image

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default = 'bmx-bumps')
    parser.add_argument("--use_crf", default = 'True')
    parser.add_argument("--sample_range", default =3)
    
    return parser.parse_args()

def configure_dataset_model(args):

    args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
    args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
    args.data_dir = '/home/xiankai/work/DAVIS-2016'   # 37572 image pairs
    args.data_list = '/home/xiankai/work/DAVIS-2016/test_seqs.txt'  # Path to the file listing the images in the dataset
    args.ignore_label = 255     #The index of the label to ignore during the training
    args.input_size = '473,473' #Comma-separated string with height and width of images
    args.num_classes = 2      #Number of classes to predict (including background)
    args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
    args.restore_from = './co_attention_davis_43.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
    args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
    args.save_segimage = True
    args.seg_save_dir = "./result/test/davis_iteration_conf_try"
    args.vis_save_dir = "./result/test/davis_vis"
    args.corp_size =(473, 473)

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it 
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can 
       load the weights file, create a new ordered dict without the module prefix, and load it back 
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new

def sigmoid(inX): 
    return 1.0/(1+np.exp(-inX))#定义一个sigmoid方法，其本质就是1/(1+e^-x)

def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)

    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = CoattentionNet(num_classes=args.num_classes, nframes = args.sample_range)
    for param in model.parameters():
        param.requires_grad = False

    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    #print(saved_state_dict.keys())
    #model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) #convert_state_dict(saved_state_dict["model"])

    model.eval()
    model.cuda()

    db_test = db.PairwiseImg(train=False, inputRes=(473,473), db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
    testloader = data.DataLoader(db_test, batch_size= 1, shuffle=False, num_workers=0)
    voc_colorize = VOCColorize()

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir) and not os.path.exists(args.vis_save_dir):
            os.makedirs(args.seg_save_dir)
            #os.makedirs(args.vis_save_dir)
    print("======> test set size:", len(testloader))
    my_index = 0
    old_temp=''
    for index, batch in enumerate(testloader):
        print('%d processd'%(index))
        target = batch['target']
        np.save('target.npy', target.float().data)
        #search = batch['search']
        temp = batch['seq_name']
        args.seq_name=temp[0]
        print(args.seq_name)
        if old_temp==args.seq_name:
            my_index = my_index+1
        else:
            my_index = 0
        output_sum = 0   
        for i in range(0,1):
            search = batch['search'+'_'+str(i)]
            search_im = search
            print('input size:', search_im.size(),len(search.size()))
            if len(search.size()) <5:
                search_im = search_im.unsqueeze(0)
            output = model(Variable(target, volatile=True).cuda(),Variable(search_im, volatile=True).cuda())
            #print(output[0]) # output有两个
            output_sum = output_sum + output[0].data[0,0].cpu().numpy() #分割那个分支的结果

            #np.save('infer'+str(i)+'.npy',output1)
            #output2 = output[1].data[0, 0].cpu().numpy() #interp'
        
        output1 = output_sum#/args.sample_range
        #target_mask = output[3].data[0,0].cpu().numpy()
        #print('output size:', output1.shape, type(output1))
        first_image = np.array(Image.open(args.data_dir+'/JPEGImages/480p/blackswan/00000.jpg'))
        original_shape = first_image.shape 
        output1 = cv2.resize(output1, (original_shape[1],original_shape[0]))
        #output2 = cv2.resize(target_mask, (original_shape[1], original_shape[0]))
        mask = (output1*255).astype(np.uint8)
        #target_mask = (output2*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        #target_mask = Image.fromarray(target_mask)

        save_dir_res = os.path.join(args.seg_save_dir, 'Results', args.seq_name)
        old_temp=args.seq_name
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)
        if args.save_segimage:
            my_index1 = str(my_index).zfill(5)
            seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
            gate_filename = os.path.join(save_dir_res, '{}_gate.png'.format(my_index1))
            mask.save(seg_filename)
            #target_mask.save(gate_filename)

if __name__ == '__main__':
    main()
