'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''

import os
import pydicom
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
import tables
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
from utils import pickle_load
import functools

def loadDCM(directory, sort = False):
    dataset = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".dcm"):
            data = pydicom.read_file(os.path.join(directory, filename))
            #data = pydicom.dcmread(os.path.join(directory, filename))
            dataset.append(data)
    
    if sort:
        dataset = sorted(dataset, key=lambda x: x.SliceLocation, reverse=True)
            
    return dataset

def loadNii(directory, expected_shape):
    img = None
    seg = None
    
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            if filename.startswith("img"): 
                img = nib.load(os.path.join(directory, filename)).get_data()
                
            elif filename.startswith("seg"): 
                seg = nib.load(os.path.join(directory, filename)).get_data()
            
    assert len(img.shape) == len(expected_shape), "Shape dimension mismatch"

    D, H, W = img.shape
    D_exp, H_exp, W_exp = expected_shape
    #print(img.shape)

    height_downsample_factor = H // H_exp
    width_downsample_factor = W // W_exp

    #pad around the img
    if H % height_downsample_factor != 0 :
        img = np.vstack((img, np.zeros((D, H_exp*(H//H_exp + 1)-H, W))))

    if W % width_downsample_factor != 0:
        D, H, W = img.shape
        img = np.hstack((img, np.zeros((D, H, W_exp*(W//W_exp +1)-W))))

    
    #Reject data with bad dimensions
    if height_downsample_factor == 0 or width_downsample_factor == 0:
        return None, None

    if H % height_downsample_factor != 0 or W % width_downsample_factor != 0:
        return None, None

    #Reject seg with invalid data
    if np.max(seg) != 1 or np.min(seg) != 0:
        return None, None

    #Downsample in H and W dimension
    img = img[:, ::height_downsample_factor, ::width_downsample_factor]
    seg = seg[:, ::height_downsample_factor, ::width_downsample_factor]

    #Pad data if D < 128
    if D < D_exp:
        img = np.vstack((img, np.zeros((D_exp-D, H_exp, W_exp))))
        seg = np.vstack((seg, np.zeros((D_exp-D, H_exp, W_exp))))    
    
    #Shape check
    assert img.shape[0] >= D_exp and img.shape[1] == H_exp and img.shape[2] == W_exp, "Patient {} img has shape {}".format(directory, img.shape)
    assert seg.shape[0] >= D_exp and seg.shape[1] == H_exp and seg.shape[2] == W_exp, "Patient {} seg has shape {}".format(directory, seg.shape)
    
    return img, seg

def crop(img, seg, expected_shape, random_crop = False):
    
    D, H, W = img.shape
    D_exp, H_exp, W_exp = expected_shape
    
    assert D >= D_exp and H == H_exp and W == W_exp
    
    if D > D_exp:

        if random_crop:
            left_ind = np.random.randint(D - D_exp)
            img = img[left_ind:left_ind + D_exp, :, :]
            seg = seg[left_ind:left_ind + D_exp, :, :]
        else:    
            numOfFramesWithTumor = sum([1 if np.sum(seg[i,:,:]) > 0 else 0 for i in range(seg.shape[0])])

            #Reject seg without tumor
            if numOfFramesWithTumor == 0:
                return None, None
            
            #print("Tumor frames:",numOfFramesWithTumor,"/",D)

            #Hack: Get most tumor-heavy 128x128x128
            best_left_ind = 0
            max_tumor_volume = 0
            if numOfFramesWithTumor <= 32:
                best_left_ind = 0
                max_tumor_volume = 0
                for i in range(D-32):
                    tumor_volume = np.sum(seg[i:i+32, :, :])
                    if tumor_volume > max_tumor_volume:
                        max_tumor_volume = tumor_volume
                        best_left_ind = i
                img = img[best_left_ind:best_left_ind+32, :, :].repeat(4, axis=0)
                seg = seg[best_left_ind:best_left_ind+32, :, :].repeat(4, axis=0)
            elif numOfFramesWithTumor <= 64:
                best_left_ind = 0
                max_tumor_volume = 0
                for i in range(D-64):
                    tumor_volume = np.sum(seg[i:i+64, :, :])
                    if tumor_volume > max_tumor_volume:
                        max_tumor_volume = tumor_volume
                        best_left_ind = i
                img = img[best_left_ind:best_left_ind+64, :, :].repeat(2, axis=0)
                seg = seg[best_left_ind:best_left_ind+64, :, :].repeat(2, axis=0)
            else:
                best_left_ind = 0
                max_tumor_volume = 0
                for i in range(D-128):
                    tumor_volume = np.sum(seg[i:i+128, :, :])
                    if tumor_volume > max_tumor_volume:
                        max_tumor_volume = tumor_volume
                        best_left_ind = i

                img = img[best_left_ind:best_left_ind+128, :, :]
                seg = seg[best_left_ind:best_left_ind+128, :, :]
    
    assert img.shape == expected_shape, "Patient img has shape {}".format(img.shape)
    assert seg.shape == expected_shape, "Patient seg has shape {}".format(seg.shape)

    #add one more dim for modality
    img = img[np.newaxis, :, :, :]  
    seg = seg[np.newaxis, :, :, :]
       
    return img, seg

#Extract 3D Pixel Array
#dataset is a list of PyDicom objects
def extractPixelMatrix(dataset):
    pixels = list(map(lambda data: data.pixel_array, dataset))
    return np.array(pixels)

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if np.random.choice([True, False]):
            axis.append(dim)
            
    return axis

# def flip_image(image, axis):
#     try:
#         new_data = np.copy(image.get_data())
#         for axis_index in axis:
#             new_data = np.flip(new_data, axis=axis_index)
#     except TypeError:
#         new_data = np.flip(image.get_data(), axis=axis)
        
#     return new_img_like(image, data=new_data)

def flip_image(image, axis):
    new_data = np.copy(image)
    for axis_index in axis:
        new_data = np.flip(new_data, axis=axis_index)
    #print("Flip:", image.shape, axis, new_data.shape)
    return new_data

# def offset_image(image, offset_factor):
#     image_data = image.get_data()
#     image_shape = image_data.shape
    
#     new_data = np.zeros(image_shape)

#     assert len(image_shape) == 3, "Wrong dimessions! Expected 3 but got {0}".format(len(image_shape))
       
#     if len(image_shape) == 3: 
#         new_data[:] = image_data[0][0][0] # 左上角背景点像素值
        
#         oz = int(image_shape[0] * offset_factor[0])
#         oy = int(image_shape[1] * offset_factor[1])
#         ox = int(image_shape[2] * offset_factor[2])
#         if oy >= 0:
#             slice_y = slice(image_shape[1]-oy)
#             index_y = slice(oy, image_shape[1])
#         else:
#             slice_y = slice(-oy,image_shape[1])
#             index_y = slice(image_shape[1] + oy)
#         if ox >= 0:
#             slice_x = slice(image_shape[2]-ox)
#             index_x = slice(ox, image_shape[2])
#         else:
#             slice_x = slice(-ox,image_shape[2])
#             index_x = slice(image_shape[2] + ox)
#         if oz >= 0:
#             slice_z = slice(image_shape[0]-oz)
#             index_z = slice(oz, image_shape[0])
#         else:
#             slice_z = slice(-oz,image_shape[0])
#             index_z = slice(image_shape[0] + oz)
#         new_data[index_z, index_y, index_x] = image_data[slice_z,slice_y,slice_x]            

#     return new_img_like(image, data=new_data)

def offset_image(image, offset_factor):
    image_shape = image.shape
    
    new_data = np.zeros(image_shape)

    assert len(image_shape) == 3, "Wrong dimessions! Expected 3 but got {0}".format(len(image_shape))
       
    if len(image_shape) == 3: 
        new_data[:] = image[0][0][0] # 左上角背景点像素值
        
        oz = int(image_shape[0] * offset_factor[0])
        oy = int(image_shape[1] * offset_factor[1])
        ox = int(image_shape[2] * offset_factor[2])
        if oy >= 0:
            slice_y = slice(image_shape[1]-oy)
            index_y = slice(oy, image_shape[1])
        else:
            slice_y = slice(-oy,image_shape[1])
            index_y = slice(image_shape[1] + oy)
        if ox >= 0:
            slice_x = slice(image_shape[2]-ox)
            index_x = slice(ox, image_shape[2])
        else:
            slice_x = slice(-ox,image_shape[2])
            index_x = slice(image_shape[2] + ox)
        if oz >= 0:
            slice_z = slice(image_shape[0]-oz)
            index_z = slice(oz, image_shape[0])
        else:
            slice_z = slice(-oz,image_shape[0])
            index_z = slice(image_shape[0] + oz)
        new_data[index_z, index_y, index_x] = image_data[slice_z,slice_y,slice_x]            

    return new_img_like(image, data=new_data)
    
def shift_and_scale_intensity(image, shift_factor):
    intensity_shift = 2.0 * np.random.random(image.shape) - 1.0 #[-1.0, 1.0]
    intensity_scale = 0.2 * np.random.random(image.shape) + 0.9 #[0.9, 1.1]
    return (image + intensity_shift * shift_factor) * intensity_scale



def augment_image(image, flip_axis=None, offset_factor=None, shift_factor=None):
    if flip_axis is not None:
        image = flip_image(image, axis=flip_axis)
    if offset_factor is not None:
        image = offset_image(image, offset_factor=offset_factor)
    if shift_factor:
        image = shift_and_scale_intensity(image, shift_factor=shift_factor)
        
    return image

def get_target_label(label_data, config):
    target_label = np.zeros(label_data.shape)
    
    for l_idx in range(config["n_labels"]):
        assert config["labels"][l_idx] in [1,2,4],"Wrong label!Expected 1 or 2 or 4, but got {0}".format(config["labels"][l_idx])
        if not config["label_containing"]:
            target_label[np.where(label_data == config["labels"][l_idx])]= 1
        else:
            if config["labels"][l_idx] == 1:
                target_label[np.where(label_data == 1)] = 1
                target_label[np.where(label_data == 4)] = 1
            elif config["labels"][l_idx] == 2:
                target_label[np.where(label_data > 0 )] = 1
            elif config["labels"][l_idx] == 4:
                target_label[np.where(label_data == 4)]= 1                
           
    return target_label
                     
class BratsDataset(Dataset):
    def __init__(self, phase, config):
        super(BratsDataset, self).__init__()
        
        self.config = config
        self.phase = phase
        self.data_name = config["data_file"]
        
        if phase == "train":
            self.data_ids = config["training_file"]
        elif phase == "validate":
            self.data_ids = config["validation_file"]
        elif phase == "test":
            self.data_ids = config["test_file"]
        
        self.data_list = pickle_load(self.data_ids)
        self.data_file = None
        
    def file_open(self):
        if self.data_file is None:
            self.data_file = open_data_file(self.data_name)
        
    def file_close(self):
        if self.data_file is not None:
            self.data_file.close()
            self.data_file = None
            
    def get_affine(self):
        return self.affine
    
    def __getitem__(self, index):
        item = self.data_list[index]
        input_data = self.data_file.root.data[item] # data shape:(4, 128, 128, 128)
        label_data = self.data_file.root.truth[item] # truth shape:(1, 128, 128, 128)
        seg_label = get_target_label(label_data, self.config)
        self.affine = self.data_file.root.affine[item]
        # dimensions of data
        n_dim = len(seg_label[0].shape)

        if self.phase == "train":
            if self.config["random_offset"]:
                offset_factor = -0.25 + np.random.random(n_dim)
            else:
                offset_factor = None
            if self.config["random_flip"]:
                flip_axis = random_flip_dimensions(n_dim)
            else:
                flip_axis = None 
                
        elif self.phase == "validate" or self.phase == "test":
            offset_factor = None
            flip_axis = None
        
        # Apply random offset and flip to each channel according to randomly generated offset factor and flip axis respectively.
        data_list = list()
        for data_channel in range(input_data.shape[0]):
            # Transform ndarray data to Nifti1Image
            channel_image = nib.Nifti1Image(dataobj=input_data[data_channel], affine=self.affine)
            data_list.append(resample_to_img(augment_image(channel_image, flip_axis=flip_axis, offset_factor=offset_factor), channel_image, interpolation="continuous").get_data())
        input_data = np.asarray(data_list)
        # Transform ndarray segmentation label to Nifti1Image
        seg_image = nib.Nifti1Image(dataobj=seg_label[0], affine=self.affine)
        seg_label = resample_to_img(augment_image(seg_image, flip_axis=flip_axis, offset_factor=offset_factor), seg_image, interpolation="nearest").get_data()
        if len(seg_label.shape) == 3:
            seg_label = seg_label[np.newaxis]
            
        if self.config["VAE_enable"]:
            # Concatenate to (5, 128, 128, 128) as network output
            final_label = np.concatenate((seg_label, input_data), axis=0)
        else:
            final_label = seg_label
            
        if self.phase == "test":
            subject_id = np.array(self.data_file.root.subject_ids[item].decode('utf-8'), dtype=str())[0]
            return input_data, final_label, subject_id
        
        return input_data, final_label
    
    def __len__(self):
        return len(self.data_list)

class StanfordDataset(Dataset):
    def __init__(self, phase, config):
        super(StanfordDataset, self).__init__()
        
        self.config = config
        self.phase = phase
        self.expected_shape = config["image_shape"]
        
        if phase == "train":
            self.data_directory = config["training_dir"]
        elif phase == "validate":
            self.data_directory = config["validation_dir"]
        elif phase == "test":
            self.data_directory = config["test_dir"]
        

        self.dataset = None
        
    def file_open(self):
        if self.dataset is None:
            self.dataset = { "images": [], "segs": [] }
            
            for subdir in os.listdir(os.fsencode(self.data_directory)):
                subdirname = os.fsdecode(subdir)
                if not subdirname.startswith("."):
                    image, seg = loadNii(os.path.join(self.data_directory, subdirname), self.expected_shape)
                    if image is None or seg is None:
                        continue
                    self.dataset["images"].append(image)
                    self.dataset["segs"].append(seg)
        
    def file_close(self):
        return
#         if self.dataset is not None:
#             self.dataset = None
            
    def get_affine(self):
        return self.affine
    
    def __getitem__(self, index):
        input_data = self.dataset["images"][index]
        seg_label = get_target_label(self.dataset["segs"][index], self.config)
        
        #n_dim should be 3
        n_dim = len(seg_label[0].shape)
        #n_dim = len(seg_label.shape)

        if self.phase == "train":
            random_crop = self.config["random_crop"]
            
            if self.config["random_offset"]:
                offset_factor = -0.25 + np.random.random(n_dim)
            else:
                offset_factor = None
            if self.config["random_flip"]:
                flip_axis = random_flip_dimensions(n_dim)
            else:
                flip_axis = None           
            if self.config["random_intensity_shift"]:
                shift_factor = 0.1
            else:
                shift_factor = None
                
                
        elif self.phase == "validate" or self.phase == "test":
            random_crop = False
            offset_factor = None
            flip_axis = None
            shift_factor = None
        
        #Data Augmentation: Currently only random crop, flip and intensity shift works 
        input_data, seg_label = crop(input_data, seg_label, self.expected_shape, random_crop=random_crop)
        #print(input_data.shape)
            
        data_list = []
        for input_channel in range(input_data.shape[0]):
            data_list.append(augment_image(
                input_data[input_channel],
                flip_axis=flip_axis, 
                offset_factor=offset_factor,
                shift_factor=shift_factor
            ))
        input_data = np.asarray(data_list)    
        seg_label  = augment_image(seg_label[0], flip_axis=flip_axis, offset_factor=offset_factor)
        if len(seg_label.shape) == 3:
            seg_label = seg_label[np.newaxis]
            
        if self.config["VAE_enable"]:
            # Concatenate to (2, 128, 128, 128) as network output
            final_label = np.concatenate((seg_label, input_data), axis=0)
        else:
            final_label = seg_label
        
        return input_data, final_label
    
    def __len__(self):
        return len(self.dataset["images"])
    
class StanfordSmallDataset(StanfordDataset):
        
    def __init__(self, phase, config, limit=5):
        super().__init__(phase, config)
        self.limit = limit 
        
    def file_open(self):
        if self.dataset is None:
            self.dataset = { "images": [], "segs": [] }
            count = 0
            for subdir in os.listdir(os.fsencode(self.data_directory)):
                if(count < self.limit):
                    subdirname = os.fsdecode(subdir)
                    if not subdirname.startswith("."):
                        print("Loading from " + subdirname)
                        image, seg = loadNii(os.path.join(self.data_directory, subdirname), self.expected_shape)
                        if image is None or seg is None:
                            continue
                        self.dataset["images"].append(image)
                        self.dataset["segs"].append(seg)
                        count += 1
                 
    def file_close(self):
        #Small dataset so no need to close
        return           



