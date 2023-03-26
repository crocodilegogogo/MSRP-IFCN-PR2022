# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:57:41 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from skimage import transform
import torch.nn.functional as F
import torch
import os

def RP_trans(x, m, tau, sign_flag, image_size):
    y = x.reshape(x.shape[0], x.shape[1], 1)
    # y = x.reshape(x.shape[1], 1)
    
    N = x.shape[1]
    N2 = N - tau*(m - 1)
    xe = []
    for mi in range(m):
        if mi == 0:
            xe = y[:, tau*mi: N2 + tau*mi, :]
        else:
            xe = np.concatenate((xe, y[:, tau*mi: N2+tau*mi, :]), axis = 2) # e.g. cut out the vector (len 2) in pos 1, 5, 9
    
    x1 = np.tile(xe, (1, N2, 1));                                           # repeat xe for N2 times
    
    xe_flat = xe.transpose(0,2,1).reshape(xe.shape[0], xe.shape[1]*xe.shape[2], 1)
    xe_flat_tile = np.tile(xe_flat, (1, 1, N2))    
    x2 = xe_flat_tile.reshape(xe.shape[0], m, N2*N2)
    x2 = x2.transpose(0,2,1)
    
    S = np.sqrt(np.sum((x1-x2)*(x1-x2), axis=2))
    
    # if sign_flag == 1:        
    #     sign_mask = np.sum((x1-x2), axis=2)
    #     sign_mask[np.where(sign_mask<0)] = -1
    #     sign_mask[np.where(sign_mask>0)] = 1
    #     S = S*sign_mask
    #     sign_mask = sign_mask.reshape(S.shape[0], N2, N2)
    
    RP_images = S.reshape(S.shape[0], N2, N2)
    RP_images = transform.resize(RP_images,(RP_images.shape[0], image_size, image_size))
    
    # use interpolate operation for image resizing
    # RP_images = F.interpolate(torch.from_numpy(RP_images).unsqueeze(0), [image_size, image_size], mode="bilinear")
    # RP_images = RP_images.squeeze()
    # if len(RP_images.shape) == 2:
    #     RP_images = RP_images.unsqueeze(0).numpy()
    # else:
    #     RP_images = RP_images.numpy()
    
    # visualization
    # for k in range(S.shape[0]):
    #     fig = plt.figure()
    #     plt.imshow(RP_images[k,:,:], cmap=plt.cm.gray)
        
    return RP_images

def RP_images(input_data, per_interval_num, m, tau, sign_flag, image_size):
    
    RP_images = []  
    
    if input_data.shape[0] <= per_interval_num:
        input_per_time = input_data[0:input_data.shape[0], :]
        RP_images = RP_trans(input_per_time, m, tau, sign_flag, image_size)    
    else:
        interval = math.ceil(input_data.shape[0]/per_interval_num)
        for i in range(interval):    
            if i != interval-1:
                input_per_time = input_data[i*per_interval_num:(i+1)*per_interval_num, :]
                if i == 0:
                    RP_images = RP_trans(input_per_time, m, tau, sign_flag, image_size)
                else:
                    RP_images = np.concatenate((RP_images, 
                                                    RP_trans(input_per_time, m, tau, sign_flag, image_size)), 
                                                  axis=0)
            else:
                input_per_time = input_data[i*per_interval_num:(input_data.shape[0]+1), :]
                RP_images = np.concatenate((RP_images, 
                                                RP_trans(input_per_time, m, tau, sign_flag, image_size)), 
                                              axis=0)            
    return RP_images

# UNIVARIATE_DATASET_NAMES = [['FiftyWords',48,3,4],['Adiac',64,2,1],['ArrowHead',64,2,1],['Beef',64,2,1],['BeetleFly',64,2,1],['BirdChicken',128,2,1],['Car',64,2,1],
# ['CBF',64,2,1],['ChlorineConcentration',112,2,1],['CinCECGTorso',128,3,4],['Coffee',64,2,1],['Computers',128,3,4],['CricketX',64,3,4],['CricketY',64,3,4],['CricketZ',64,3,4],
# ['DiatomSizeReduction',64,2,1],['DistalPhalanxOutlineAgeGroup',72,2,1],['DistalPhalanxOutlineCorrect',64,2,1],['DistalPhalanxTW',64,2,1],['Earthquakes',64,2,1],
# ['ECG200',64,2,1],['ECG5000',64,2,1],['ECGFiveDays',64,3,4],['ElectricDevices',88,2,1],['FaceAll',96,2,1],['FaceFour',96,2,1],['FacesUCR',96,2,1],['FISH',64,2,1],
# ['FordA',64,2,1],['FordB',128,3,4],['GunPoint',64,2,1],['Ham',64,3,4],['HandOutlines',128,3,4],['Haptics',64,2,1],['Herring',64,3,4],['InlineSkate',128,2,1],
# ['InsectWingbeatSound',48,3,4],['ItalyPowerDemand',16,2,1],['LargeKitchenAppliances',128,2,1],['Lightning2',64,3,4],['Lightning7',64,3,4],['MALLAT',128,3,4],
# ['Meat',64,2,1],['MedicalImages',96,2,1],['MiddlePhalanxOutlineAgeGroup',64,3,4],['MiddlePhalanxOutlineCorrect',64,3,4],['MiddlePhalanxTW',64,3,4],['MoteStrain',80,2,1],
# ['NonInvasiveFetalECGThorax1',128,3,4],['NonInvasiveFetalECGThorax2',128,3,4],['OliveOil',96,2,1],['OSULeaf',96,2,1],['PhalangesOutlinesCorrect',64,2,1],
# ['Phoneme',128,3,4],['Plane',64,3,4],['ProximalPhalanxOutlineAgeGroup',64,2,1],['ProximalPhalanxOutlineCorrect',64,2,1],['ProximalPhalanxTW',64,3,4],['RefrigerationDevices',128,3,4],
# ['ScreenType',128,3,4],['ShapeletSim',64,3,4],['ShapesAll',64,2,1],['SmallKitchenAppliances',128,3,4],['SonyAIBORobotSurface1',64,2,1],['SonyAIBORobotSurface2',64,2,1],
# ['StarLightCurves',128,3,4],['Strawberry',64,2,1],['SwedishLeaf',64,2,1],['Symbols',64,3,4],['SyntheticControl',64,2,1],['ToeSegmentation1',64,3,4],['ToeSegmentation2',64,3,4],
# ['Trace',64,2,1],['TwoLeadECG',64,2,1],['TwoPatterns',64,2,1],['UWaveGestureLibraryAll',128,3,4],['UWaveGestureLibraryX',64,3,4],['UWaveGestureLibraryY',64,3,4],
# ['UWaveGestureLibraryZ',64,2,1],['Wafer',64,2,1],['Wine',64,3,4],['WordSynonyms',64,2,1],['Worms',128,3,4],['WormsTwoClass',128,3,4],['Yoga',64,2,1]]

UNIVARIATE_DATASET_NAMES = [['Coffee',64,2,1]]

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
(data_file_path, _) = os.path.split(CUR_DIR)

for dataset_param in UNIVARIATE_DATASET_NAMES:
    
    dataset_id = dataset_param[0]
    image_size = dataset_param[1]
    m = dataset_param[2]
    tau = dataset_param[3]    
    
    train = pd.read_csv(data_file_path+'//archives/UCR_Archive//'+dataset_id+'/'+dataset_id+'_TRAIN.tsv',
                        sep='\t',header = None,
                        engine='python')
    train = np.array(train)
    train_x = train[:,1:]
    
    test = pd.read_csv(data_file_path+'//archives/UCR_Archive//'+dataset_id+'/'+dataset_id+'_TEST.tsv',
                       sep='\t',header = None,
                       engine='python')
    test = np.array(test)
    test_x = test[:,1:]

    plt.close('all')
    
    save_path = data_file_path+'//archives//RP_UCR_Archive//'+dataset_id+'/'
    if os.path.exists(save_path):
        print(dataset_id+': already exist, pass!') 
        continue         
    else: 
        os.makedirs(save_path)
    
    # RP_images(input_data, interval_num, m, tau, sign_flag, image_size) # the num of samples per interval

    RP_imgs_train_store = RP_images(train_x, 50, m, tau, 0, image_size)
    RP_imgs_test_store  = RP_images(test_x, 50, m, tau, 0, image_size)
    
    # visualization
    # for k in range(RP_imgs_train_store.shape[0]):
    #     fig = plt.figure()
    #     plt.imshow(RP_imgs_train_store[k,:,:], cmap=plt.cm.gray)    
    
    np.save(save_path+dataset_id+'_TRAIN.npy', RP_imgs_train_store)
    print(dataset_id+',train_set save over')
    np.save(save_path+dataset_id+'_TEST.npy', RP_imgs_test_store)
    print(dataset_id+',test_set save over')

# try_train = np.load(save_path+dataset_id+'_TRAIN.npy')
# try_test = np.load(save_path+dataset_id+'_TEST.npy')