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
import torchvision
import torch.nn.functional as F

class Inception(nn.Module):
    
    def __init__(self, in_channels, out1_channels_1, out1_channels_2, out1_channels_3, 
                 out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3, 
                 out2_channels_4, num_class):
        super(Inception,self).__init__()
        
        in_channels_module1 = in_channels
        self.module1 = Inception_module(in_channels, out1_channels_1, out1_channels_2, 
                                        out1_channels_3, out1_channels_4)
        
        in_channels_module2 = out1_channels_1 + out1_channels_2 + out1_channels_3 + out1_channels_4
        self.module2 = Inception_module(in_channels_module2, out2_channels_1, out2_channels_2, 
                                        out2_channels_3, out2_channels_4)
        
        in_channels_module3 = out2_channels_1 + out2_channels_2 + out2_channels_3 + out2_channels_4
        self.module3 = Inception_module(in_channels_module3, out1_channels_1, out1_channels_2, 
                                        out1_channels_3, out1_channels_4)
            
        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels_module2, num_class)
        
    def forward(self, x):
        
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_ave_pooling(x).squeeze()
        
        output = self.linear(x)
        
        return output, x

class Inception_module(nn.Module):
    
    def __init__(self, in_channels, out_channels_1, out_channels_2, 
                 out_channels_3, out_channels_4, **kwargs):
        super(Inception_module,self).__init__()        
        
        self.conv_1 = BasicConv2d(in_channels, out_channels_1, 1)
        
        self.conv_1_3_1 = BasicConv2d(in_channels, out_channels_2, 1)
        self.conv_1_3_3 = BasicConv2d(out_channels_2, out_channels_2, 3)
        
        self.conv_1_3_3_1 = BasicConv2d(in_channels, out_channels_3, 1)
        self.conv_1_3_3_31 = BasicConv2d(out_channels_3, out_channels_3, 3)
        self.conv_1_3_3_32 = BasicConv2d(out_channels_3, out_channels_3, 3)
        
        self.conv_3pool_1_3 = nn.AvgPool2d(3, 1, 3//2)
        self.conv_3pool_1_1 = BasicConv2d(in_channels, out_channels_4, 1)
        
    def forward(self, x):
        
        branch_1 = self.conv_1(x)
        
        branch_2 = self.conv_1_3_1(x)
        branch_2 = self.conv_1_3_3(branch_2)
        
        branch_3 = self.conv_1_3_3_1(x)
        branch_3 = self.conv_1_3_3_31(branch_3)
        branch_3 = self.conv_1_3_3_32(branch_3)
        
        branch_4 = self.conv_3pool_1_3(x)
        branch_4 = self.conv_3pool_1_1(branch_4)
        
        module_outputs = [branch_1, branch_2, branch_3, branch_4]
        
        return torch.cat(module_outputs, 1)
        

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
def train_op(classifier_obj, EPOCH, batch_size, LR, train_x, train_y, 
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
    optimizer = torch.optim.Adam(classifier_obj.parameters(),lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                              patience=50, 
                                              min_lr=0.0001, verbose=True)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    
    # save init model    
    output_directory_init = output_directory_models+'init_model.pkl'
    torch.save(classifier_obj.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    for epoch in range (EPOCH):
        
        for step, (x,y) in enumerate(train_loader):
               
            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc = classifier_obj(batch_x)[0]
            
            # cal the sum of pre loss per batch 
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test per epoch
        classifier_obj.eval()
        loss_train, accuracy_train = get_test_loss_acc(classifier_obj, loss_function, train_x, train_y, test_split)        
        loss_test, accuracy_test = get_test_loss_acc(classifier_obj, loss_function, test_x, test_y, test_split) 
        classifier_obj.train()  
       
        
        # update lr
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']
        
        ######################################dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(classifier_obj.eval(), loss_function, train_x, train_y, test_split)
        
        # loss_test, accuracy_test = get_loss_acc(classifier_obj.eval(), loss_function, test_x, test_y, test_split)
        
        # classifier_obj.train()
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
        
        training_duration_logs = save_models(classifier_obj, output_directory_models, 
                                             loss_train, loss_train_results, 
                                             accuracy_test, accuracy_test_results, 
                                             model_save_interval, epoch, EPOCH, 
                                             start_time, training_duration_logs, 
                                             save_best_train_model, save_best_test_model)
    
    # save last_model
    output_directory_last = output_directory_models+'last_model.pkl'
    torch.save(classifier_obj.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_test_results, accuracy_test_results)    
    
    return(history, training_duration_logs)

