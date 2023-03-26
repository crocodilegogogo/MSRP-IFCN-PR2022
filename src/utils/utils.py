import numpy as np
import pandas as pd 
import matplotlib
import math

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os,shutil
import operator
import torch
import torch.utils.data as Data
import time 

from utils.constants import parse_args
args = parse_args()
DATASET_NAMES = [args.UNIVARIATE_DATASET_NAMES]
from utils.constants import create_classifier
import sklearn 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def check_if_file_exits(file_name):
    return os.path.exists(file_name)

# load RP data and the data labels
def readucr(filename, filename_RP, delimiter=','):
    
    data = pd.read_csv(filename,sep='\t',header = None,
                       engine='python')
    data = np.array(data)
    X = np.load(filename_RP)
    Y = data[:,0]
    
    return X, Y

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

def create_path(root_dir,classifier_name, archive_name):
    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+'/'
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

# load the training and test data (labels), and put them to datasets_dict
def read_all_datasets(root_dir, archive_name, RP_archive_name, split_val = False):

    datasets_dict = {}

    dataset_names_to_sort = []

    for dataset_name in DATASET_NAMES:
        root_dir_dataset =root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        root_dir_dataset_RP =root_dir+'/archives/'+RP_archive_name+'/'+dataset_name+'/'
        file_name = root_dir_dataset + dataset_name
        file_name_RP = root_dir_dataset_RP + dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN.tsv', file_name_RP+'_TRAIN.npy')
        x_test, y_test = readucr(file_name+'_TEST.tsv', file_name_RP+'_TEST.npy')
        # if split val from train, save val dataset to a novel csv
        if split_val == True:
            # check if dataset has already been splitted
            temp_dir =root_dir_dataset+'TRAIN_VAL/'
            # print(temp_dir)
            train_test_dir = create_directory(temp_dir)
            # print(train_test_dir)
            if train_test_split is None:
                # then do no re-split because already splitted
                # read train set
                x_train,y_train = readucr(temp_dir+dataset_name+'_TRAIN')
                # read val set
                x_val,y_val = readucr(temp_dir+dataset_name+'_VAL')
            else:
                # split for cross validation set
                x_train,x_val,y_train,y_val  = train_test_split(x_train,y_train,
                    test_size=0.25)
                # concat train set
                train_set = np.zeros((y_train.shape[0],x_train.shape[1]+1),dtype=np.float64)
                train_set[:,0] = y_train
                train_set[:,1:] = x_train
                # concat val set
                val_set = np.zeros((y_val.shape[0],x_val.shape[1]+1),dtype=np.float64)
                val_set[:,0] = y_val
                val_set[:,1:] = x_val
                # save the train set
                np.savetxt(temp_dir+dataset_name+'_TRAIN.tsv',train_set,delimiter=',')
                # save the val set
                np.savetxt(temp_dir+dataset_name+'_VAL.tsv',val_set,delimiter=',')

            # put val dataset to datasets_dict
            datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_val.copy(),
                y_val.copy(),x_test.copy(),y_test.copy())
            
        # only put training and test datasets to datasets_dict
        else:
            datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())

        dataset_names_to_sort.append((dataset_name,len(x_train)))

    dataset_names_to_sort.sort(key=operator.itemgetter(1))

    for i in range(len(DATASET_NAMES)):
        DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

# obtain the training and test data, make the min labels to zeros
def prepare_data(datasets_dict, dataset_name):
    
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test  = datasets_dict[dataset_name][2]
    y_test  = datasets_dict[dataset_name][3]
    
    # obtain the total number of categories
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min labels to zeros
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignaly because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)

    if len(x_train.shape) == 3:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    
    return x_train, y_train, x_test, y_test, y_true, y_true_train, nb_classes

# generate the directories for saving models
def generate_output_directory(iter, root_dir, classifier_name, archive_name, dataset_name):
    
    if iter==0:
        trr = ''
    else:
        trr = '_iter_'+str(iter)
    # ./results/FCN_torch/UCR_Archive_TSC_iter_i(i:0,1,2,3,4)/
    tmp_output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+trr+'/'
    # ./results/FCN_torch/UCR_Archive_TSC_iter_i(i:0,1,2,3,4)/Coffee/
    output_directory        = tmp_output_directory+dataset_name+'/'
    output_directory_models = output_directory+'saved_models/'
    # if classifier_name!='nne' and classifier_name!='ensembletransfer':
    temp_output_directory        = create_directory(output_directory)
    temp_output_directory_models = create_directory(output_directory_models)

    return output_directory, output_directory_models

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None : 
        # index for when resplitting the concatenation 
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit 
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder，让encoder知道类别数从小到大的排列情况
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels 
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test，把调整好的label按照index重新放回去
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test 
    else: 
        # no validation split 
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit 
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder 
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels 
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

# the training process of the classifier
def fit_classifier(classifier_name, EPOCH, BATCH_SIZE, LR, 
                   x_train, y_train, x_test, y_test, nb_classes,
                   output_directory_models, output_directory,
                   model_save_interval, test_split, 
                   save_best_train_model, save_best_test_model):
    
    classifier, classifier_func = create_classifier(classifier_name, nb_classes)
    classifier.cuda()
    print(classifier)
    classifier_parameter = get_parameter_number(classifier)
  
    history, models_time_logs = classifier_func.train_op(classifier, EPOCH, BATCH_SIZE, LR, 
                                                         x_train, y_train, x_test, y_test, 
                                                         output_directory_models, model_save_interval, 
                                                         test_split)
    model_save_ids = np.arange(model_save_interval,EPOCH+1, model_save_interval).tolist()
    # model_save_ids.append(EPOCH) # id:EPOCH model included    
    load_models_save_log(classifier, EPOCH, x_test, y_test, nb_classes,
                         output_directory_models, output_directory, 
                         model_save_ids, history, 
                         models_time_logs, test_split,
                         save_best_train_model, 
                         save_best_test_model)
    return history, models_time_logs

# obtain the parameter number of the classifier
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Network_Total_Parameters:', total_num, 'Network_Trainable_Parameters:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}

# obtain the logging directory of the classifier
def obtain_result_logging_directory(classifier_name, ITERATIONS_RANGE, root_dir):
    
    ensemble_name = get_ensemble_name(classifier_name, ITERATIONS_RANGE)
    mean_name = get_mean_name(classifier_name, ITERATIONS_RANGE)
    print('classifier_name:', classifier_name, 'will be ensembled as', ensemble_name)
    # the output path of all datasets' accuracies
    output_directory = root_dir+'/results/'+ensemble_name+'/'
    create_directory(output_directory)
    # the file of all datasets acc logs
    classifier_best_train_logname = output_directory+ensemble_name+'_best_train_logs.csv'
    classifier_best_test_logname = output_directory+ensemble_name+'_best_test_logs.csv'
    classifier_best_train_mean_logname = output_directory+mean_name+'_best_train_logs.csv'
    classifier_best_test_mean_logname = output_directory+mean_name+'_best_test_logs.csv'
    
    return output_directory, classifier_best_train_logname, classifier_best_test_logname,\
           classifier_best_train_mean_logname, classifier_best_test_mean_logname

# calculate the test loss and accs
def get_test_loss_acc(net, loss_function, x_data, y_data, test_split=1):
    loss_sum_data = torch.tensor(0)
    true_sum_data = torch.tensor(0)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = x_data.shape[0] // test_split,
                                    shuffle = False)  
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()            
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()               
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]    
    return loss, acc

# perform model inference
def model_predict(net, x_data, y_data, test_split=1):
    predict = [] 
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = x_data.shape[0] // test_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # y = y.cuda()
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc.cpu().data.numpy()
            pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            pre = pred_bc.cpu().data.numpy()
            if pre.size == 1:
                pre = pre.tolist()
                predict.append(pre)
            else:
                pre = pre.tolist()
                predict.extend(pre)  
            output.extend(out)
    acc = sum(predict == y_data)/y_data.shape[0]
    return predict, output , acc

# calculate the test metrics
def calculate_metrics(y_true, y_pred, duration, nb_classes, y_true_val=None,y_pred_val=None): 
    
    # metrics
    res = pd.DataFrame(data = np.zeros((1,4),dtype=np.float), index=[0], 
        columns=['precision','accuracy','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)
    save_models
    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    
    # confusion matrixes
    confusion_matrixes = confusion_matrix(y_true, y_pred) # rows true label
    confusion_matrixes = pd.DataFrame(confusion_matrixes)
    
    # false predictions
    false_index = np.where(np.array(y_true)!=np.array(y_pred))[0].tolist()
    y_correct = np.array(y_true)[np.array(y_true)!=np.array(y_pred)].tolist()
    pre_false = np.array(y_pred)[np.array(y_true)!=np.array(y_pred)].tolist()
    false_pres = pd.DataFrame(data = np.zeros((len(false_index),3),dtype=np.int64), 
                              columns=['index','real_category','predicted_category'])
    false_pres['index'] = false_index
    false_pres['real_category'] = y_correct
    false_pres['predicted_category'] = pre_false    
    
    return res, confusion_matrixes, false_pres

# log the training and test histories
def log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                loss_test_results, accuracy_test_results):
    
    history = pd.DataFrame(data = np.zeros((EPOCH,5),dtype=np.float), 
                           columns=['train_acc','train_loss','lr','test_acc','test_loss'])
    history['train_acc']  = accuracy_train_results
    history['train_loss'] = loss_train_results
    history['lr']         = lr_results
    history['test_acc']   = accuracy_test_results
    history['test_loss']  = loss_test_results
    
    return history

# if reach the model-saving intervals or the last epoch, save the models!
def save_models(net, output_directory_models, loss_train, loss_train_results, 
                accuracy_test, accuracy_test_results, 
                model_save_interval, epoch, EPOCH, 
                start_time, training_duration_logs,
                save_best_train_model, save_best_test_model):   
    
    output_directory_best_train = output_directory_models+'best_train_model.pkl'
    if loss_train <= min(loss_train_results):        
        torch.save(net.state_dict(), output_directory_best_train)
        
    output_directory_best_test = output_directory_models+'best_test_model.pkl'
    if accuracy_test >= max(accuracy_test_results):        
        torch.save(net.state_dict(), output_directory_best_test)
    
    if (epoch+1) % model_save_interval == 0 or (epoch+1) == EPOCH:
        
        if save_best_train_model:
            # get the part before '.pkl' of output_directory_best_train
            flag_name_train = output_directory_best_train.split('.',1)[0]
            model_id = str(epoch+1)
            # Add the 'model_id' number after the original model name 
            new_directory_train = flag_name_train + '_' + model_id +'.pkl'
            try:
                shutil.copyfile(output_directory_best_train, new_directory_train)
                print('copy best_train_model as best_train_model_'+model_id)
            except:
                print('copy failure in Epoch_'+model_id+' for best train model')
                
        if save_best_test_model:
            # get the part before '.pkl' of output_directory_best_test
            flag_name_test = output_directory_best_test.split('.',1)[0]
            model_id = str(epoch+1)
            # Add the 'model_id' number after the original model name
            new_directory_test = flag_name_test + '_' + model_id +'.pkl'
            # rename the original model name
            try:                
                shutil.copyfile(output_directory_best_test, new_directory_test)
                print('copy best_test_model as best_test_model_'+model_id)
            except:
                print('copy failure in Epoch_'+model_id+' for best test model')
        # log training time 
        training_duration = time.time() - start_time
        training_duration_logs.append(training_duration)   
        
    return(training_duration_logs)

# save test logs
def save_logs(output_directory, hist_df, model_id, flag_name, y_pred, y_true, 
              nb_classes, duration, lr=True, y_true_val=None, y_pred_val=None):

    df_metrics, confusion_matrixes, false_pres = calculate_metrics(y_true, y_pred, duration,
                                                                   y_true_val,y_pred_val,
                                                                   nb_classes)
    df_path = output_directory+'df_metrics.csv'
    flag_df_path = df_path.split('.',1)[0]
    flag_new_df_path = flag_df_path + '_' + flag_name +'_' + str(model_id) +'.csv'
    df_metrics.to_csv(flag_new_df_path, index=False)    
    
    # history[:model_id,:]
    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    index_best_model = hist_df['train_loss'].idxmin() 
    row_best_model = hist_df.loc[index_best_model]   
    df_best_model['best_model_train_loss'] = row_best_model['train_loss']
    df_best_model['best_model_test_loss'] = row_best_model['test_loss']
    df_best_model['best_model_train_acc'] = row_best_model['train_acc']
    df_best_model['best_model_test_acc'] = row_best_model['test_acc']  
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model    
    df_best_model.to_csv(flag_new_df_path, index=False, mode='a+')
    
    confusion_matrixes.to_csv(flag_new_df_path, index=True, mode='a+')
    false_pres.to_csv(flag_new_df_path, index=False, mode='a+')

    # return df_metrics

# plot the training histories
def plot_epochs_metric(hist, file_name, metric):
    plt.figure()
    metric_train = 'train_'+metric
    metric_test = 'test_'+metric
    plt.plot(hist[metric_train], label = metric_train)
    plt.plot(hist[metric_test], label = metric_test)
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend([metric_train, metric_test], loc='upper left')
    plt.savefig(file_name,dpi=300,bbox_inches='tight')
    plt.close()

# load the models, perform inference and save logs
def load_models_save_log(network, EPOCH, test_x, test_y, nb_classes,
                            output_directory_models, output_directory,
                            model_save_ids, history, training_duration_logs, 
                            test_split, save_best_train_log = True,
                            save_best_test_log = True):
    
    # load saved models, predict, cal metrics and save logs    
    history.to_csv(output_directory+'history.csv', index=False)
    # plot losses 
    plot_epochs_metric(history, output_directory+'epochs_loss.png', 'loss')
    
    for step, model_id in enumerate(model_save_ids):
        
        if save_best_train_log:
            # generate train objects
            best_train_net = network
            # load best train saved models
            train_model_directory = output_directory_models+'best_train_model_'+str(model_id)+'.pkl'
            best_train_net.load_state_dict(torch.load(train_model_directory))
            best_train_net.eval()
            # get prediction of best train saved models
            predict_best_train_net,_,_ = model_predict(best_train_net.cuda(), test_x, test_y, test_split)
            print(str(model_id),':',str(sum(predict_best_train_net == test_y)/test_y.shape[0]))
            save_logs(output_directory, history, model_id, 'best_train',
                      predict_best_train_net, test_y, nb_classes,
                      training_duration_logs[step])
            
        if save_best_test_log:
            # generate test objects
            best_test_net = network  
            # load best test saved models
            test_model_directory = output_directory_models+'best_test_model_'+str(model_id)+'.pkl'
            best_test_net.load_state_dict(torch.load(test_model_directory))
            best_test_net.eval()        
            # get prediction of best test saved models
            predict_best_test_net,_,_ = model_predict(best_test_net.cuda(), test_x, test_y, test_split)
            print(str(model_id),':',str(sum(predict_best_test_net == test_y)/test_y.shape[0]))
            save_logs(output_directory, history, model_id, 'best_test',
                      predict_best_test_net, test_y, nb_classes,
                      training_duration_logs[step])

# generate the name of the ensemble log file
def get_ensemble_name(classifier_name, ITERATIONS_RANGE):
    
    ITERATIONS_RANGE_str = str(ITERATIONS_RANGE)
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace('[','')
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace(']','')
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace(',','_')
    ensemble_name        = classifier_name+'_ensemble_'+ITERATIONS_RANGE_str
    
    return ensemble_name

# generate the name of the mean log file
def get_mean_name(classifier_name, ITERATIONS_RANGE):
    
    ITERATIONS_RANGE_str = str(ITERATIONS_RANGE)
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace('[','')
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace(']','')
    ITERATIONS_RANGE_str = ITERATIONS_RANGE_str.replace(',','_')
    mean_name = classifier_name+'_mean_'+ITERATIONS_RANGE_str
    
    return mean_name

# save the ensembled or averaged classification results of all datasets, epoches and classifiers to a csv file
def save_ensemble_or_mean_results_to_csv(classifier_best_logname, dataset_name, log_cur_dataset_epoch_accs):
    
    if os.path.isfile(classifier_best_logname):
        read_history_files = pd.read_csv(classifier_best_logname,header=0,index_col=0)
        # if dataset have not be saved, save it
        if not dataset_name in read_history_files.index:
            concat_flag = pd.concat([read_history_files,log_cur_dataset_epoch_accs])
            concat_flag.to_csv(classifier_best_logname)
    else:
        log_cur_dataset_epoch_accs.to_csv(classifier_best_logname)

# 
def classier_predict_and_save_results(flag_train_test, x_test, y_test, nb_classes, network,
                                      model_save_interval, EPOCH, dataset_name, ITERATIONS_RANGE,
                                      classifier_name, root_dir, archive_name, test_split,
                                      output_dataset_directory, classifier_best_logname, 
                                      classifier_best_mean_logname):
    # which epoch models will be tested
    model_epoch_test_ids = np.arange(model_save_interval,EPOCH, model_save_interval).tolist()
    model_epoch_test_ids.append(EPOCH) # id:EPOCH model included
    # log_cur_dataset_epoch_accs:log the accs of different epoches
    model_epoch_test_indexes = []
    for i in range(len(model_epoch_test_ids)):
        model_epoch_test_indexes.append('Epoch_'+str(model_epoch_test_ids[i])) # eg. Epoch_500
    log_cur_dataset_epoch_accs = pd.DataFrame(data = np.zeros((1,len(model_epoch_test_ids)),dtype=np.float), 
                                              index=[dataset_name], columns = model_epoch_test_indexes)        
    log_cur_dataset_epoch_mean_accs = pd.DataFrame(data = np.zeros((1,len(model_epoch_test_ids)),dtype=np.float), 
                                                   index=[dataset_name], columns = model_epoch_test_indexes)        
    # cal the averaged outputs of different ITERATIONS in different epoches, notice that those are not category probabilities
    start_curdataset_time = time.time()
    # iter_mean_epoches_concat_output: log the outputs of all iter(sum together) and epoches(log in columns)
    iter_mean_epoches_concat_output = np.zeros((len(model_epoch_test_ids)*y_test.shape[0], nb_classes))
    iter_mean_epoches_add_accs = np.zeros((1, len(model_epoch_test_ids)))                    
    # start to cal the averaged values of all iters
    for iter in ITERATIONS_RANGE:
        print('\t\t',classifier_name,'best_' + flag_train_test +':iter_',iter)        
        trr = ''
        if iter!=0:
            trr = '_iter_'+str(iter)
        # e.g.: ./results/FCN_torch/UCR_Archive_TSC_iter_i(i:0,1,2,3,4)/Coffee/saved_models/
        # find saved model's path
        saved_model_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+trr+'/'+dataset_name+'/'+'saved_models'+'/'               
        # cur_iter_best_model_output: log the outputs of current iter(sum together) and epoches(log in columns)
        cur_iter_best_model_output = []
        cur_iter_best_model_acc    = []
        # concat the prediction results of different epoches
        # cycle saved epoch models
        for step, model_id in enumerate(model_epoch_test_ids):
            # generate network objects     
            network_obj = network
            # load best saved models
            train_model_directory = saved_model_directory+'best_'+flag_train_test+'_model_'+str(model_id)+'.pkl'
            network_obj.load_state_dict(torch.load(train_model_directory))
            network_obj.eval()
            with torch.no_grad():
                # get outputs of best saved models by concat them, get the acc of every Epoch_id       
                _,output_best_model,acc_best_model = model_predict(network_obj.cuda(), x_test, y_test, test_split)
            cur_iter_best_model_output.extend(output_best_model)
            cur_iter_best_model_acc.append(acc_best_model)
        # sum the outputs of all iterations one by one
        iter_mean_epoches_concat_output = iter_mean_epoches_concat_output + np.array(cur_iter_best_model_output)
        # sum the accs of all iterations one by one 
        iter_mean_epoches_add_accs = iter_mean_epoches_add_accs + np.array(cur_iter_best_model_acc)
    # average the outputs of all iterations
    iter_mean_epoches_concat_output = iter_mean_epoches_concat_output/len(ITERATIONS_RANGE)
    # obtain the final predictions of all iterations
    iter_mean_epoches_concat_predictions = np.argmax(iter_mean_epoches_concat_output ,1).tolist()
    # average the accs of all iterations
    iter_mean_epoches_acc = iter_mean_epoches_add_accs/len(ITERATIONS_RANGE)
    iter_mean_epoches_acc = iter_mean_epoches_acc[0]
    # cal current dataset all iter and epoch test time(epoch should not be caled, mean them)
    duration = time.time() - start_curdataset_time
    # iter_mean_epoches_concat_predictions is the concatenated results of all epoches, now split the results of each epoch and save them to different csv files seperately
    for step, model_index in enumerate(model_epoch_test_indexes):                        
        iter_mean_epoch_prediction = iter_mean_epoches_concat_predictions[step*y_test.shape[0] : (step+1)*y_test.shape[0]]
        df_metrics, confusion_matrixes, false_pres = calculate_metrics(y_test, iter_mean_epoch_prediction, 
                                                                duration/len(model_epoch_test_ids), 
                                                                nb_classes)
        # save the metrics of current dataset
        df_path = output_dataset_directory+'df_metrics.csv'
        flag_df_path = df_path.split('.',1)[0]
        flag_new_df_path = flag_df_path + '_best_'+flag_train_test+'_' + model_index +'.csv'
        df_metrics.to_csv(flag_new_df_path, index=False)
        confusion_matrixes.to_csv(flag_new_df_path, index=True, mode='a+')
        false_pres.to_csv(flag_new_df_path, index=False, mode='a+')
        # log current dataset ensemble acc
        log_cur_dataset_epoch_accs[model_index] = df_metrics['accuracy'][0]
        log_cur_dataset_epoch_mean_accs[model_index] = iter_mean_epoches_acc[step]
        
    save_ensemble_or_mean_results_to_csv(classifier_best_logname, dataset_name, 
                                         log_cur_dataset_epoch_accs)
    
    save_ensemble_or_mean_results_to_csv(classifier_best_mean_logname, dataset_name, 
                                         log_cur_dataset_epoch_mean_accs)