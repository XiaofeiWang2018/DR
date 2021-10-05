from os import listdir
from os.path import join
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
from skimage import io,transform
import random
import torch
import platform
import numpy as np
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt
import cv2
sysstr = platform.system()
from sklearn.neighbors import NearestNeighbors
import tarfile
from os import remove
from os.path import exists, join, basename
from transform.transforms_group import RandomCrop
from torchvision.transforms import Compose,  ToTensor, ToPILImage, CenterCrop, Resize


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
    ])
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class Dataset_seg(data.Dataset):
    def __init__(self, image_paths, mask_paths,upscale_factor,blur_kernel,noise_var,init_size,lesion,transform=None):
        super(Dataset_seg, self).__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.upscale_factor = upscale_factor
        self.blur_kernel = blur_kernel
        self.noise_var = noise_var
        self.transform=transform
        self.masks = {'MA': [], 'HM': [], 'HE': [], 'SE': []}
        self.images = []

        for image_path, MA_path, HM_path, HE_path, SE_path in zip(image_paths, mask_paths['MA'], mask_paths['HM'],
                                                                  mask_paths['HE'], mask_paths['SE']):
            self.images.append(image_path)
            self.masks['MA'].append(MA_path)
            self.masks['HM'].append(HM_path)
            self.masks['HE'].append(HE_path)
            self.masks['SE'].append(SE_path)

        self.lesion = lesion
        self.init_size = init_size

        if transform is not None:
            self.rotation_angle=transform['rotation_angle']
            self.Crop_factor=transform['Crop_factor']
            self.lr_transform = train_lr_transform(int(self.init_size / self.Crop_factor), self.upscale_factor)
            self.hr_transform = train_hr_transform(int(self.init_size / self.Crop_factor))

        else:
            self.transform =None
            self.Crop_factor=1

    def __getitem__(self, index):

        info = [Image.open(self.images[index])]
        info.append(Image.open(self.masks['MA'][index]))
        info.append(Image.open(self.masks['HM'][index]))
        info.append(Image.open(self.masks['HE'][index]))
        info.append(Image.open(self.masks['SE'][index]))
        masks = []


        if self.transform is not None:  # train
            info = self.hr_transform(info)
            hr_image=info[0]
            lr_image = self.lr_transform(hr_image)
            hr_scale = Resize(int(self.init_size / self.Crop_factor), interpolation=Image.BICUBIC)
            hr_restore_img = hr_scale(lr_image)
            mask_all = np.zeros(shape=(int(1024 // self.Crop_factor), int(1024 // self.Crop_factor)), dtype=np.float64)

        else:
            hr_image = CenterCrop(self.init_size)(info[0])
            lr_scale = Resize(self.init_size // self.upscale_factor, interpolation=Image.BICUBIC)
            lr_image = lr_scale(hr_image)
            hr_scale = Resize(self.init_size, interpolation=Image.BICUBIC)
            hr_restore_img = hr_scale(lr_image)
            mask_all = np.zeros(shape=(1024, 1024), dtype=np.float64)


        for j in range(1,5):
            info[j] = np.array(info[j])
            info[j][info[j] > 80] = 255
            info[j][info[j] <= 80] = 0
            info[j] = Image.fromarray(info[j].astype('uint8'))
        if self.lesion['MA']:
            MA_mask = np.array(np.array(info[1]))[:, :, 0]/255.0
            masks.append(MA_mask)
            mask_all = mask_all + MA_mask

        if self.lesion['HM']:
            HM_mask = np.array(np.array(info[2]))[:, :, 0] /255.0
            masks.append(HM_mask)
            mask_all = mask_all + HM_mask

        if self.lesion['HE']:
            HE_mask = np.array(np.array(info[3]))[:, :, 0] /255.0
            masks.append(HE_mask)
            mask_all = mask_all + HE_mask

        if self.lesion['SE']:
            SE_mask = np.array(np.array(info[4]))[:, :, 0] /255.0
            masks.append(SE_mask)
            mask_all = mask_all + SE_mask
        if self.lesion['BG']:
            BG_mask = 1 - mask_all
            masks.append(BG_mask)

        masks=np.transpose(np.array(masks),(1,2,0))
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image), ToTensor()(masks)

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path,if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))


            return img.convert('RGB')




def get_set_patch_seg(upscale_factor,blur_kernel,noise_var,lesion,dataset='DDR',init_size=1024,mode='Train',transform=None):
    phase=mode
    imgs_ori = glob.glob('../data/' + dataset + '/Segmentation/Original_Images/' + phase + '/img_after2/*.jpg')
    imgs_ori.sort()
    imgs_ori_list = listdir('../data/' + dataset + '/Segmentation/Original_Images/' + phase + '/img_after2')
    imgs_ori_list.sort()
    mask_paths = {'MA': [], 'HM': [], 'HE': [], 'SE': []}
    image_paths = imgs_ori
    for i in range(len(imgs_ori)):

        if os.path.exists('../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Microaneurysms/img_after/' +imgs_ori_list[i]):
            mask_paths['MA'].append(
                '../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Microaneurysms/img_after/' +
                imgs_ori_list[i])
        if os.path.exists('../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Haemorrhages/img_after/' +imgs_ori_list[i]):
            mask_paths['HM'].append(
                '../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Haemorrhages/img_after/' +
                imgs_ori_list[i])
        if os.path.exists('../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Hard_Exudates/img_after/' +imgs_ori_list[i]):
            mask_paths['HE'].append(
                '../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Hard_Exudates/img_after/' +
                imgs_ori_list[i])
        if os.path.exists('../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Soft_Exudates/img_after/' +imgs_ori_list[i]):
            mask_paths['SE'].append(
                '../data/' + dataset + '/Segmentation/Groundtruths/' + phase + '/Soft_Exudates/img_after/' +
                imgs_ori_list[i])

    return Dataset_seg(image_paths, mask_paths,upscale_factor,blur_kernel,noise_var,init_size,lesion,transform=transform)



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""-------------------------------------------------------  ---stage2-----------------------------------------"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class DataLoader_cls(data.Dataset):
    def __init__(self, cls_image_paths,root_img_cls,root_label_cls,upscale_factor,blur_kernel,noise_var,init_size,transform=None):
        super(DataLoader_cls, self).__init__()
        self.cls_image_paths = cls_image_paths
        self.root_img_cls = root_img_cls
        self.root_label_cls = root_label_cls
        self.upscale_factor = upscale_factor
        self.blur_kernel = blur_kernel
        self.noise_var = noise_var
        self.transform=transform

        self.init_size = init_size
        self.transform =None
        self.Crop_factor=1

    def __getitem__(self, index):

        info = [Image.open(self.cls_image_paths[index])]
        hr_image = CenterCrop(self.init_size)(info[0])
        lr_scale = Resize(self.init_size // self.upscale_factor, interpolation=Image.BICUBIC)
        lr_image = lr_scale(hr_image)
        hr_scale = Resize(self.init_size, interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)

        full_label_path = self.root_label_cls + self.cls_image_paths[index][len(self.root_img_cls) + 1:-3] + 'txt'
        label_gt = open(full_label_path, 'r').readline()
        label = int(label_gt[0])

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image), torch.from_numpy(np.array(label)).long()

    def __len__(self):
        return len(self.cls_image_paths)




def get_set_patch_cls(upscale_factor,blur_kernel,noise_var,dataset='DDR',init_size=1024,mode='Train',transform=None):
    phase=mode


    #  classification
    imgs_ori_cls = glob.glob('../data/' + dataset + '/Grading/Original_Images/' + phase + '/img_before_ori/*.jpg')
    imgs_ori_cls.sort()
    cls_image_paths = imgs_ori_cls
    root_img_cls = '../data/' + dataset + '/Grading/Original_Images/' + phase + '/img_before_ori'
    root_label_cls = '../data/' + dataset + '/Grading/Groundtruths/' + phase + '/'


    return DataLoader_cls(cls_image_paths,root_img_cls,root_label_cls,upscale_factor,blur_kernel,noise_var,init_size,transform=transform)


if __name__ == '__main__':
    lesion = {'MA': True, 'HM': True, 'HE': True, 'SE': True, 'BG': True}
    train_set = get_set_patch(8,(3,3),0.001,lesion, dataset='DDR', init_size=1024
                                   ,transform={'Crop_factor':4,'rotation_angle':0})


    a,c,v,vb=train_set.getitem__(0)
    a=1





