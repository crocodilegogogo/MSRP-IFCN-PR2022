import argparse

# UNIVARIATE_DATASET_NAMES = ['FiftyWords','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','Car','CBF','ChlorineConcentration','CinCECGTorso','Coffee',
# 'Computers','CricketX','CricketY','CricketZ','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW',
# 'Earthquakes','ECG200','ECG5000','ECGFiveDays','ElectricDevices','FaceAll','FaceFour','FacesUCR','FISH','FordA','FordB','GunPoint','Ham','HandOutlines',
# 'Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lightning2','Lightning7','MALLAT','Meat','MedicalImages',
# 'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OliveOil',
# 'OSULeaf','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices',
# 'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface1','SonyAIBORobotSurface2','StarLightCurves','Strawberry','SwedishLeaf','Symbols',
# 'SyntheticControl','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','TwoPatterns','UWaveGestureLibraryAll','UWaveGestureLibraryX','UWaveGestureLibraryY',
# 'UWaveGestureLibraryZ','Wafer','Wine','WordSynonyms','Worms','WormsTwoClass','Yoga']

# UNIVARIATE_DATASET_NAMES = ['Coffee']

# ITERATIONS = 1 # nb of random runs for random initializations

# # UNIVARIATE_ARCHIVE_NAMES = ['MSRP_UCR_Archive','RP_UCR_Archive','TRP_UCR_Archive']
# UNIVARIATE_ARCHIVE_NAMES = ['MSRP_UCR_Archive']

# # CLASSIFIERS = ['FCN_torch','ResNet_torch', 'Inception_torch', 'IFCN_torch', 'IRN_torch', 'AlexNet_torch', 'CNN_torch']
# CLASSIFIERS = ['IFCN_torch']

def parse_args():
      # The training options
      parser = argparse.ArgumentParser(description='MSRP-IFCN')
      parser.add_argument('--flag_train_or_test', type=str, default='Train',
                          help='pattern: Train, Test')
      parser.add_argument('--ITERATIONS', type=int, default=1,
                          help='train the model for ITERATIONS times')
      parser.add_argument('--UNIVARIATE_ARCHIVE_NAMES', type=str, default='MSRP_UCR_Archive',
                          help='archive name: MSRP_UCR_Archive, RP_UCR_Archive, TRP_UCR_Archive')
      parser.add_argument('--UNIVARIATE_DATASET_NAMES', type=str, default='Coffee',
                          help='dataset name: can be all of the 85 UCR dataset names')
      parser.add_argument('--CLASSIFIERS', type=str, default='IFCN_torch',
                          help='classifier name: FCN_torch, ResNet_torch, Inception_torch, IFCN_torch, IRN_torch, AlexNet_torch, CNN_torch')
      parser.add_argument('--BATCH_SIZE', type=int, default=16,
                          help='training batch size: can be adjusted according to different datasets')
      parser.add_argument('--EPOCH', type=int, default=1500,
                          help='training epoches')
      parser.add_argument('--model_save_interval', type=int, default=500,
                          help='save model at model_save_interval*i epoches')
      parser.add_argument('--LR', type=float, default=0.00005,
                          help='learning Rate')
      parser.add_argument('--test_split', type=int, default=8,
                          help='the testing dataset is seperated into test_split pieces in the inference process')
      parser.add_argument('--save_best_train_model', type=bool, default='True',
                          help='save_best_train_model: True, False')
      parser.add_argument('--save_best_test_model', type=bool, default='True',
                          help='save_best_test_model: True, False')
      parser.add_argument('--read_best_train_model', type=bool, default='True',
                          help='read_best_train_model: True, False')
      parser.add_argument('--read_best_test_model', type=bool, default='True',
                          help='read_best_test_model: True, False')
      args = parser.parse_args()
      return args

def create_classifier(classifier_name, nb_classes):
        
    if classifier_name=='FCN_torch': 
        from classifiers import FCN_torch 
        # __init__(self,input_channel, kernel_size1, kernel_size2, kernel_size3, 
        #          feature_channel1, feature_channel2, 
        #          feature_channel3, num_class)
        return FCN_torch.FCN(1, 8, 5, 3, 128, 
                             256, 128, nb_classes), FCN_torch
    
    if classifier_name=='ResNet_torch': 
        from classifiers import ResNet_torch 
        # __init__(self, net_layers, kernel_size1, kernel_size2, kernel_size3, 
        #          cross_layer, feature_channel_num, num_class)
        return ResNet_torch.ResNet(9, 7, 5, 3, 3, 64, nb_classes), ResNet_torch
    
    if classifier_name=='Inception_torch': 
        from classifiers import Inception_torch 
        # __init__(self, in_channels, out1_channels_1, out1_channels_2, out1_channels_3, 
        #          out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3, 
        #          out2_channels_4, num_class)
        return Inception_torch.Inception(1, 16, 32, 96, 16, 32, 64, 192, 32, nb_classes), Inception_torch
    
    if classifier_name=='IFCN_torch': 
        from classifiers import IFCN_torch 
        # __init__(self, in_channels, out1_channels_1, out1_channels_2, out1_channels_3, 
        #          out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3, 
        #          out2_channels_4, num_class)
        return IFCN_torch.IFCN(1, 16, 32, 96, 16, 32, 64, 192, 32, nb_classes), IFCN_torch
    
    if classifier_name=='IRN_torch': 
        from classifiers import IRN_torch 
        # __init__(self, block_num, in_channels, out1_channels_1, out1_channels_2, out1_channels_3, 
        #          out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3, out2_channels_4, block_num, num_class)
        return IRN_torch.IRN(3, 1, 16, 32, 96, 16, 32, 64, 192, 32, nb_classes), IRN_torch
    
    if classifier_name=='AlexNet_torch': 
        from classifiers import AlexNet_torch 
        # __init__(self,input_channel, num_class)
        return AlexNet_torch.AlexNet(1, nb_classes), AlexNet_torch
    
    if classifier_name=='CNN_torch': 
        from classifiers import CNN_torch 
        # __init__(self,input_channel, num_class)
        return CNN_torch.CNN(1, nb_classes), CNN_torch