#General imports
import argparse
import glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pydicom
from subprocess import call
import sys

parser = argparse.ArgumentParser(description='Converts DICOM to NifTI.')

parser.add_argument('--src', action='store', dest='src_dir', help='The source directory', required=True)
parser.add_argument('--out', action='store', dest='out_dir', help='The output directory', required=True)
parser.add_argument('--patient', action='store', dest='patient', help='The patient ID', required=True)
parser.add_argument('--sweeps', type=int, default=1, dest='num_sweep', help='The number of NifTI files to divide into', required=False)
parser.add_argument('--transpose', action='store_true', help='Flag to transpose the 2D MRIs')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

def loadDCM(directory, sort = False):
    dataset = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".dcm"):
            data = pydicom.read_file(os.path.join(directory, filename))
            dataset.append(data)
    
    if sort:
        #dataset = sorted(dataset, key=lambda x: x.SliceLocation, reverse=True)
        dataset = sorted(dataset, key=lambda x: x.InstanceNumber, reverse=True)
            
    return dataset

def displayDCMInfo(dataset, start = 0, end = -1, verbose = False):
    
    if verbose:
        for i in range(start, end):
            print(dataset[i])
            print()
    else:
        #Print common attributes once
        with dataset[0] as data:
            print("Patient id..............:", data.PatientID)
            print("Modality................:", data.Modality)
            print("Study Description.......:", data.StudyDescription)
            print("Study Date..............:", data.StudyDate)
            print("Series Description......:", data.SeriesDescription)
            print("Series Instance UID.....:", data.SeriesInstanceUID)
            print("Frame of Reference UID..:", data.FrameOfReferenceUID)
            if 'PixelData' in data:
                rows = int(data.Rows)
                cols = int(data.Columns)
                print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                    rows=rows, cols=cols, size=len(data.PixelData)))
                if 'PixelSpacing' in data:
                    print("Pixel spacing....:", data.PixelSpacing)
            print()
        
        for i in range(start, end):
            data = dataset[i]
            print("SOP Instance UID........:", data.SOPInstanceUID)
            print("Instance Number.........:", data.InstanceNumber)
            if 'SliceLocation' in dataset[i]:
                print("Image Position.....:", data.ImagePositionPatient)
                print("Image Orientation..:", data.ImageOrientationPatient)
                print("Slice Location.....:", data.SliceLocation)
            print()

        
#Extract 3D Pixel Array
def extractPixelMatrix(dataset):
    pixels = list(map(lambda data: data.pixel_array, dataset))
    return np.array(pixels)

def saveAsNifti(data, directory, patient):
    directory = os.path.join(directory, patient)
    try:
        os.makedirs(directory)
    except FileExistsError:
        None
    data = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(data, os.path.join(directory, "volume-" + patient + ".nii"))
    
def main(args):

    # Read from DICOM file
    dataset = loadDCM(args.src_dir, sort = True) 

    # Extract 3D Pixel Array
    dataset_pixels = extractPixelMatrix(dataset)
    
    assert dataset_pixels.ndim == 3, "Pixel matrix not 3D: " + str(dataset_pixels.shape)

    #Transpose (Optional)
    if args.transpose:
        dataset_pixels = dataset_pixels.transpose(0, 2, 1)

    D, H, W = dataset_pixels.shape
    
    if args.num_sweep == 1:
        print("3D image dimension:", (D, H, W))
        saveAsNifti(dataset_pixels, args.out_dir, args.patient)
    else:    

        D = D // args.num_sweep
        
        print("3D image dimension:", (D, H, W))

        #Save as .nii file
        for i in range(args.num_sweep):
            saveAsNifti(dataset_pixels[D*i:D*(i+1),:,:], args.out_dir, args.patient + '_' + str(i))
        
if __name__ == "__main__":
    assert args.num_sweep > 0
    main(args)
