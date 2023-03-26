# from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
# from utils.constants import CLASSIFIERS
# from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
# from utils.constants import ITERATIONS

from utils.utils import *
from utils.constants import *
# from utils.constants import create_classifier

import numpy as np
import pandas as pd
import sklearn
import time 
import os
import torch
import gc
import torch.utils.data as Data
# import torchvision

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
(root_dir, _) = os.path.split(CUR_DIR)

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
    
    if args.flag_train_or_test == 'Train':
        for classifier_name in [args.CLASSIFIERS]:
            print('classifier_name',classifier_name)
            for archive_name in [args.UNIVARIATE_ARCHIVE_NAMES]:
                print('\t RParchive_name:', archive_name)
                datasets_dict = read_all_datasets(root_dir, 'UCR_Archive', archive_name)
                # repeat the training process for ITERATIONS times
                for iter in range(args.ITERATIONS):
                    print('\t\t',classifier_name,':iter_',iter)
                    for dataset_name in [args.UNIVARIATE_DATASET_NAMES]:
                        print('\t\t\tdataset_name: ', dataset_name)
                        x_train, y_train, x_test, y_test, y_true, y_true_train, nb_classes = prepare_data(datasets_dict, dataset_name)
                        # generate_output_directory
                        output_directory, output_directory_models = generate_output_directory(iter, root_dir, classifier_name, archive_name, dataset_name)
                        # training process
                        history, training_time = fit_classifier(classifier_name, args.EPOCH, args.BATCH_SIZE, args.LR, 
                                                                x_train, y_train, x_test, y_test, nb_classes,
                                                                output_directory_models, output_directory, args.model_save_interval, 
                                                                args.test_split, args.save_best_train_model, args.save_best_test_model)
                        print('\t\t\t\t TRAINING DONE')
    
    if args.flag_train_or_test == 'Test':
        
        ITERATIONS_RANGE = np.arange(args.ITERATIONS).tolist()
        
        # cycle classifiers
        for classifier_name in [args.CLASSIFIERS]:
            # obtain the directories for logging results
            output_directory, classifier_best_train_logname, classifier_best_test_logname,\
            classifier_best_train_mean_logname, classifier_best_test_mean_logname\
               = obtain_result_logging_directory(classifier_name, ITERATIONS_RANGE, root_dir)
            
            for archive_name in [args.UNIVARIATE_ARCHIVE_NAMES]:
                print('\tRParchive_name',archive_name)
                datasets_dict = read_all_datasets(root_dir, 'UCR_Archive', archive_name)
                
                for dataset_name in [args.UNIVARIATE_DATASET_NAMES]:
                    print('\t\t\tdataset_name: ', dataset_name)
                    x_train, y_train, x_test, y_test, y_true, y_true_train, nb_classes = prepare_data(datasets_dict, dataset_name)
                    network, classifier_func = create_classifier(classifier_name, nb_classes)
                    # the output path of per dataset's metric
                    output_dataset_directory = output_directory+archive_name+'/'+dataset_name+'/'
                    create_directory(output_dataset_directory)
                    
                    if args.read_best_train_model:
                        classier_predict_and_save_results('train', x_test, y_test, nb_classes, network,
                                                          args.model_save_interval, args.EPOCH, dataset_name, ITERATIONS_RANGE,
                                          classifier_name, root_dir, archive_name, args.test_split,
                                          output_dataset_directory, classifier_best_train_logname,
                                          classifier_best_train_mean_logname)
                    
                    if args.read_best_test_model:
                        classier_predict_and_save_results('test', x_test, y_test, nb_classes, network,
                                                          args.model_save_interval, args.EPOCH, dataset_name, ITERATIONS_RANGE,
                                          classifier_name, root_dir, archive_name, args.test_split,
                                          output_dataset_directory, classifier_best_test_logname,
                                          classifier_best_test_mean_logname) 
                    
                    print('\t\t\t\t TESTING DONE')
                    

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    gc.collect()
    torch.cuda.empty_cache()