#General imports
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
# from pydicom.data import get_testdata_files
# from pydicom.filereader import read_dicomdir
from subprocess import call
import sys
import xml.etree.ElementTree as ET

#Pytorch imports
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler


#Local imports
from utils import Logger,load_old_model
from train import train_epoch
from validation import val_epoch
from nvnet import MiniNvNet, NvNet
from metrics import CombinedLoss, SoftDiceLoss
from dataset import BratsDataset, StanfordDataset, StanfordSmallDataset


#GPU Parameters/Status
#os.environ["CUDA_VISIBLE_DEVICES"]="1,0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

torch.cuda.empty_cache()
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')

# call(["nvcc", "--version"]) does not work
# nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

DATASET_GLOBAL_MEAN = 321.56370587244527
DATASET_GLOBAL_STDDEV = 517.4083720223107

DATASET_SAG_MEAN = 319.38926782103283
DATASET_SAG_STDDEV = 447.42789129337154

DATASET_AXIAL_MEAN = 337.1649891681073
DATASET_AXIAL_STDDEV = 632.1473853506837

#Config
config = dict()
config["cuda_devices"] = True
# config["labels"] = (1, 2, 4)
config["labels"] = (1,) # change label to train
config["model_file"] = os.path.abspath("single_label_{}_dice.h5".format(config["labels"][0]))
config["initial_learning_rate"] = 1e-4 #1e-5
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["image_shape"] = (128, 128, 128)
# config["image_shape"] = (160, 512, 512)
# config["image_shape"] = (160, 256, 256)
config["activation"] = "relu"
config["normalizaiton"] = "group_normalization"
config["mode"] = "trilinear"
config["n_labels"] = len(config["labels"])
#config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["all_modalities"] = ["bilat"]#["t1", "sag"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["loss_k1_weight"] = 0.3
config["loss_k2_weight"] = 0.3
config["random_offset"] = False # Boolean. Augments the data by randomly move an axis during generating a data
config["random_flip"] = True  # Boolean. Augments the data by randomly flipping an axis during generating a data
# config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["random_crop"] = True  # Boolean. Augments the data by randomly cropping in depth dimension
config["random_intensity_shift"] = True  # Boolean. Augments the data by randomly shifting and scaling pixel values
config["result_path"] = "./checkpoint_models/"
config["data_file"] = os.path.abspath("isensee_mixed_brats_data.h5")
# config["training_file"] = os.path.abspath("isensee_mixed_training_ids.pkl")
# config["validation_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
# config["test_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
config["training_dir"] = "data/preprocessed/axial/train"
config["validation_dir"] = "data/preprocessed/axial/val"
config["test_dir"] = "data/preprocessed/axial/test"
config["saved_model_file"] = None#"checkpoint_models/single_label_1_dice/epoch_199_val_loss_1.3715_acc_0.2401.pth"
config["overwrite"] = False  # If True, will create new files. If False, will use previously written files.
config["L2_norm"] = 1e-5
config["patience"] = 30
config["lr_decay"] = 0.9
config["epochs"] = 400
config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
config["VAE_enable"] = True  # Boolean. If True, will enable the VAE module.

#Detect GPU availability
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    config["cuda_devices"] = True
else:
    device = torch.device('cpu')
    config["cuda_devices"] = None

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

#Definition for train cycle
def main():
    # init or load model
    print("init model with input shape",config["input_shape"])
    model = NvNet(config=config)
    #model = MiniNvNet(config=config)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay = config["L2_norm"])
    start_epoch = 1

    if config["VAE_enable"]:
        loss_function = CombinedLoss(k1=config["loss_k1_weight"], k2=config["loss_k2_weight"])
    else:
        loss_function = SoftDiceLoss()
    # data_generator
    print("data generating")
    training_data = StanfordDataset(phase="train", config=config)
    validation_data = StanfordDataset(phase="validate", config=config)
#     training_data = StanfordSmallDataset(phase="train", config=config, limit=5)
#     validation_data = StanfordSmallDataset(phase="validate", config=config, limit=1)

    train_logger = Logger(model_name=config["model_file"],header=['epoch', 'loss', 'acc', 'lr'])

    if config["cuda_devices"] is not None:
        #gpu_list = list(range(0, 2))
        #model = nn.DataParallel(model, gpu_list)  # multi-gpu training
        model = model.cuda()
        loss_function = loss_function.cuda() 
#         model = model.to(device=device)  # move the model parameters to CPU/GPU
#         loss_function = loss_function.to(device=device)
        
    if not config["overwrite"] and config["saved_model_file"] is not None:
        if not os.path.exists(config["saved_model_file"]):
            raise Exception("Invalid model path!")
        model, start_epoch, optimizer = load_old_model(model, optimizer, saved_model_path=config["saved_model_file"])    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"],patience=config["patience"])
    
    #model = torch.load("checkpoint_models/run0/best_model_file_24.pth")
    
    print("training on label:{}".format(config["labels"]))
    max_val_acc = 0.
    for i in range(start_epoch,config["epochs"]):
        train_epoch(epoch=i, 
                    data_set=training_data, 
                    model=model,
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    logger=train_logger) 
        
        val_loss, val_acc = val_epoch(epoch=i, 
                  data_set=validation_data, 
                  model=model, 
                  criterion=loss_function, 
                  opt=config,
                  optimizer=optimizer, 
                  logger=train_logger)
        scheduler.step(val_loss)
        if config["checkpoint"] and val_acc >= max_val_acc - 0.10:#0.01:
            max_val_acc = val_acc
            save_dir = os.path.join(config["result_path"], config["model_file"].split("/")[-1].split(".h5")[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_states_path = os.path.join(save_dir,'epoch_{0}_val_loss_{1:.4f}_acc_{2:.4f}.pth'.format(i, val_loss, val_acc))
            states = {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_states_path)
            save_model_path = os.path.join(save_dir, "best_model_file_{0}.pth".format(i))
            if os.path.exists(save_model_path):
                os.system("rm "+save_model_path)
            torch.save(model, save_model_path)

if __name__ == "__main__":
    main()
