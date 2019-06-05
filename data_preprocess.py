import functools
import glob
import nibabel as nib
import numpy as np
import os
import sys

def calculateStats(directory):
    
    voxelSum = 0.0
    voxelSumSq = 0.0
    numVoxels = 0
    
    maxVal = float('-inf')
    maxFile = None
    minVal = float('inf')
    minFile = None
    
    for subdir in os.listdir(os.fsencode(directory)):
        subdirname = os.fsdecode(subdir)
        if not subdirname.startswith("."):
            full_subdir_path = os.path.join(directory, subdirname)
            for file in os.listdir(os.fsencode(full_subdir_path)):
                filename = os.fsdecode(file)
                if filename.endswith(".nii"):
                    if filename.startswith("volume"): 
                        full_file_path = os.path.join(full_subdir_path, filename)
                        img = nib.load(full_file_path).get_data()
                        voxelSum += np.sum(img)
                        voxelSumSq += np.sum(np.square(img))
                        numVoxels += img.shape[0] * img.shape[1] * img.shape[2]
                        ma = np.max(img)
                        if ma > maxVal:
                            maxVal = ma
                            maxFile = full_file_path
                        mi = np.min(img)
                        if mi < minVal:
                            minVal = mi  
                            minFile = full_file_path
    
    mean = voxelSum / numVoxels
    stddev = (voxelSumSq / numVoxels - mean**2)**(0.5)
                    
    return mean, stddev, minVal, maxVal, minFile, maxFile

def rebuildNii(directory, folder_name, mean, stddev):
    img = None
    final_seg = None
    segs = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            if filename.startswith("volume"): 
                img = nib.load(os.path.join(directory, filename)).get_data()
                
            elif filename.startswith("SEG"): 
                seg = nib.load(os.path.join(directory, filename)).get_data()
                seg = seg[:,:,:,0]                
                segs.append(seg)
                
    if(len(segs) == 0):
        final_seg = np.zeros(img.shape)
    elif(len(segs) == 1):
        final_seg = segs[0]
    else:
        final_seg = functools.reduce(lambda a, b: np.bitwise_or(a, b), segs)
            
    D, H, W = img.shape
    
    #hack to move depth to 1st dim
    if D == H:
        img = img.transpose(2, 0, 1)
        final_seg = final_seg.transpose(2, 0, 1)
        D, W = W, D
        
    #normalize image
    img = (img - mean) / stddev
        
    final_img = nib.Nifti1Image(img, affine=np.eye(4))
    final_seg_img = nib.Nifti1Image(final_seg, affine=np.eye(4))

    os.makedirs(folder_name)
    nib.save(final_seg_img, os.path.join(folder_name, "seg.nii"))
    nib.save(final_img, os.path.join(folder_name, "img.nii"))

    
def main():
    directory = 'data/raw/sag'

    #Calculate training data mean and stddev
    mean, stddev, minVal, maxVal, minFile, maxFile = calculateStats(directory + '/train')
    print(mean, stddev, minVal, maxVal, minFile, maxFile)

    #Calculated values
    DATASET_GLOBAL_MEAN = 321.56370587244527
    DATASET_GLOBAL_STDDEV = 517.4083720223107

    DATASET_SAG_MEAN = 319.38926782103283
    DATASET_SAG_STDDEV = 447.42789129337154

    #Preprocess all data
    for subdir in os.listdir(os.fsencode(directory)):
        subdirname = os.fsdecode(subdir)
        if not subdirname.startswith("."):
            path1 = os.path.join(directory, subdirname)
            for subsubdir in os.listdir(os.fsencode(path1)):
                subsubdirname = os.fsdecode(subsubdir)
                if not subsubdirname.startswith("."):
                    path2 = os.path.join(path1, subsubdirname)
                    newPath = path2.replace("raw", "preprocessed")
                    rebuildNii(path2, newPath, mean, stddev)
                    
if __name__ == "__main__":
    main()