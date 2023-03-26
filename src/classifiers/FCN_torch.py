import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from utils.utils import create_directory
from utils.utils import get_test_loss_acc
from utils.utils import save_models
from utils.utils import log_history
from utils.utils import calculate_metrics
from utils.utils import save_logs
from utils.utils import model_predict
from utils.utils import plot_epochs_metric
import os

class FCN(nn.Module):
    def __init__(self,input_channel, kernel_size1, kernel_size2, kernel_size3, 
                 feature_channel1, feature_channel2, 
                 feature_channel3, num_class):
        super(FCN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, feature_channel1, kernel_size1, 1, kernel_size1//2),
            nn.BatchNorm2d(feature_channel1),
            nn.ReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel1, feature_channel2, kernel_size2, 1, kernel_size2//2),
            nn.BatchNorm2d(feature_channel2),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel2, feature_channel3, kernel_size3, 1, kernel_size3//2),
            nn.BatchNorm2d(feature_channel3),
            nn.ReLU(),
            )
        
        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(feature_channel3, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_ave_pooling(x).squeeze()

        output = self.linear(x)
        return output, x
    
def train_op(fcn, EPOCH, batch_size, LR, train_x, train_y, 
             test_x, test_y, output_directory_models, 
             model_save_interval, test_split, 
             save_best_train_model = True,
             save_best_test_model = True):
    # prepare training_data
    BATCH_SIZE = int(min(train_x.shape[0]/10, batch_size))
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    drop_last = drop_last_flag
                                   )
    
    # init lr&train&test loss&acc log
    lr_results = []
    loss_train_results = []    
    accuracy_train_results = []
    loss_test_results = []    
    accuracy_test_results = []    
    
    # prepare optimizer&scheduler&loss_function
    optimizer = torch.optim.Adam(fcn.parameters(),lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                              patience=50, 
                                              min_lr=0.0001, verbose=True)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    
    # save init model    
    output_directory_init = output_directory_models+'init_model.pkl'
    torch.save(fcn.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    for epoch in range (EPOCH):
        
        for step, (x,y) in enumerate(train_loader):
               
            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc = fcn(batch_x)[0]
            
            # cal the sum of pre loss per batch 
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test per epoch
        fcn.eval()
        loss_train, accuracy_train = get_test_loss_acc(fcn, loss_function, train_x, train_y, test_split)
        loss_test, accuracy_test = get_test_loss_acc(fcn, loss_function, test_x, test_y, test_split) 
        fcn.train()  
       
        
        # update lr
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']
        # print(lr)
        
        ######################################dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(fcn.eval(), loss_function, train_x, train_y, test_split)
        
        # loss_test, accuracy_test = get_loss_acc(fcn.eval(), loss_function, test_x, test_y, test_split)
        
        # fcn.train()
        ##################################################################################
        
        # log lr&train&test loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)    
        accuracy_train_results.append(accuracy_train)
        loss_test_results.append(loss_test)    
        accuracy_test_results.append(accuracy_test)
        
        # print training process
        if (epoch+1) % 10 == 0:
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train, 
                  '| train_acc:', accuracy_train, 
                  '| test_loss:', loss_test, 
                  '| test_acc:', accuracy_test)
        
        training_duration_logs = save_models(fcn, output_directory_models, 
                                             loss_train, loss_train_results, 
                                             accuracy_test, accuracy_test_results, 
                                             model_save_interval, epoch, EPOCH, 
                                             start_time, training_duration_logs, 
                                             save_best_train_model, save_best_test_model)        
        
        
    
    # save last_model
    output_directory_last = output_directory_models+'last_model.pkl'
    torch.save(fcn.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_test_results, accuracy_test_results)    
    
    return(history, training_duration_logs)

