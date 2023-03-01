# usage:
# $ cd your_GRSS_dir

import os
import math
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np

if __name__ == '__main__':
    data_root ="/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/"
    src_img_dir = data_root + 'train/'
    src_rgb_dir = src_img_dir + 'rgb'
    src_sar_dir = src_img_dir + 'sar'
    src_dsm_dir = src_img_dir + 'dsm'
    
    train_img_dir = data_root + 'train_temp/'
    train_rgb_dir = train_img_dir + 'rgb'
    train_sar_dir = train_img_dir + 'sar'
    train_dsm_dir = train_img_dir + 'dsm'
    
    val_img_dir = data_root + 'val_temp/'
    val_rgb_dir = val_img_dir + 'rgb'
    val_sar_dir = val_img_dir + 'sar'
    val_dsm_dir = val_img_dir + 'dsm'
    

    for dir in [train_img_dir, train_rgb_dir, train_sar_dir, train_dsm_dir,\
                val_img_dir,   val_rgb_dir,   val_sar_dir,   val_dsm_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    filenames = os.listdir(src_rgb_dir)
    num_files = len(filenames)
    print('total file nums:', num_files)#1773

    train_ratio = 0.9
    num_train = int(num_files * train_ratio)
    all_index = range(num_files)
    train_index = range(num_train)

    for idx in tqdm(all_index):
        image_id = filenames[idx][:-4]
        src_rgb = src_img_dir +'/rgb/'+ image_id + '.tif'
        src_sar = src_img_dir +'/sar/'+ image_id + '.tif'
        src_dsm = src_img_dir +'/dsm/'+ image_id + '.tif'
        if idx in train_index:
            out_rgb_path = train_rgb_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_rgb, out_rgb_path))
            
            out_sar_path = train_sar_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_sar, out_sar_path))
            
            out_dsm_path = train_dsm_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_dsm, out_dsm_path))
        else:
            out_rgb_path = val_rgb_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_rgb, out_rgb_path))
            
            out_sar_path = val_sar_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_sar, out_sar_path))
            
            out_dsm_path = val_dsm_dir + '/' + image_id + '.tif'
            os.system('cp {} {}'.format(src_dsm, out_dsm_path))
            
    print("train_rgb: {}, val_rgb: {}".format(num_train, num_files - num_train)) # train_rgb: 2342, val_rgb: 586
    