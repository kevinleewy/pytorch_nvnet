#General imports
import argparse
import glob
import nibabel as nib
import numpy as np
import os
import sys

#Pytorch imports
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

#Local imports
from nvnet import NvNet


parser = argparse.ArgumentParser(description='Performs 3D segmentation of Breast MRI.')

parser.add_argument('--src', action='store', dest='src', help='The source file', required=True)
parser.add_argument('--out', action='store', dest='out_dir', help='The output directory', required=True)
parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='Saved checkpoint', required=False, default="checkpoint_models/v1/axial/best_model_file_356.pth")
parser.add_argument('--no-vae', action='store_true', help='Flag to disable VAE')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

def loadConfig(args):
    config = dict()
    config["cuda_devices"] = None
    config["VAE_enable"] = not args.no_vae  # Boolean. If True, will enable the VAE module.
    config["best_model_file"] = "checkpoint_models/v1/axial/best_model_file_356.pth"
    config["best_model_file"] = args.checkpoint
    config["input_file"] = args.src

    return config

def loadModel(config, USE_GPU = True):
    dtype = torch.float32 # we will be using float throughout this tutorial

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        config["cuda_devices"] = True
    else:
        device = torch.device('cpu')
        config["cuda_devices"] = None

    #Load model
    model = torch.load(config["best_model_file"], map_location=device)
    model.eval()
    return model

def segment(model, config):

    #Load input
    img = padBadDim(nib.load(config["input_file"]).get_data())
    D, H, W = img.shape
    print(img.shape)
    height_downsample_factor = H//128
    width_downsample_factor = W//128

    
    final_output = np.zeros((D, 128, 128))

    if D >= 128:

        summed_output = np.zeros((D, 128, 128))
        denom = np.zeros((D, 1))

        for i in range(D-128+1):
            inputs = img[np.newaxis, np.newaxis, i:i+128, ::height_downsample_factor, ::width_downsample_factor]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.type(torch.FloatTensor)
            if config["cuda_devices"] is not None:
                inputs = inputs.cuda()
            with torch.no_grad():
                if config["VAE_enable"]:
                    outputs, distr = model(inputs)
                else:
                    outputs = model(inputs)   
            output_array = np.asarray(outputs.tolist())
            output_array = output_array[0,0,:,:,:]
            summed_output[i:i+128, :, :] += output_array
            denom[i:i+128] += 1

        for i in range(D):
            final_output[i, :, :] = summed_output[i, :, :] / denom[i]

    else:
        inputs = np.vstack((img[:, ::height_downsample_factor, ::width_downsample_factor], np.zeros((128-D, 128, 128))))
        inputs = inputs[np.newaxis, np.newaxis, :, :, :]
        inputs = torch.from_numpy(inputs)
        inputs = inputs.type(torch.FloatTensor)
        if config["cuda_devices"] is not None:
            inputs = inputs.cuda()
        with torch.no_grad():
            if config["VAE_enable"]:
                outputs, distr = model(inputs)
            else:
                outputs = model(inputs)   
        output_array = np.asarray(outputs.tolist())
        final_output = output_array[0,0,:D,:,:]
        
    #Upsample width and height dimension
    final_output = final_output.repeat(height_downsample_factor, axis=1).repeat(width_downsample_factor, axis=2)

    #Convert to binary mask
    #final_output = np.where(final_output > 0.5, 1, 0)

    return final_output


def padBadDim(img):
    #pad around the img
    D,H,W = img.shape
    
    if H % 128 != 0:
        newDim = np.zeros((D, 128*(H//128 + 1)-H, W))
        img = np.hstack((img, newDim))

    if W % 128 != 0:
        D, H, W = img.shape
        newDim = np.zeros((D, H, 128*(W//128 + 1)-W))
        img = np.concatenate((img, newDim),axis=2)
    
    return img
        

def saveAsNifti(data, directory, filename="prediction"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(data, os.path.join(directory, filename + ".nii"))
    
def main(args):
    config = loadConfig(args)
    model = loadModel(config)
    segmentation = segment(model, config)

    #Save prediction
    saveAsNifti(segmentation, args.out_dir, "probabilities")
        
if __name__ == "__main__":
    main(args)

