from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from numpy import *
import torchvision.utils as utils
# from network import AlexNet, resnet34, AlexNet_CE_MSE
# from data_process import DRDataset,DRDataset_AC
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
from network import U_Net_Cut
from network import MICCAI17_Generator as Generator
from network import Discriminator as Discriminator
from util import LambdaLR
from torchvision.utils import save_image
from metrics import SSIM,computeAUPR
from torch.utils.data import DataLoader
from seg_loss import FocalLoss, DiceLoss
from data_process import get_set_patch,display_transform
from math import log10
from matplotlib import pyplot as plt
import scipy.misc
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_ssim
import pandas as pd

if 1:
    dataset = 'DDR'
    mode='Train'
    seg_mode='hr'
    upscale_factor = 4
    Crop_factor = 2

lesion = {'MA': True, 'HM': True, 'HE': True, 'SE': True, 'BG': True}
device_ids = [0]
img_dir=  'train_size256_up4' if mode=='Train' else 'test_size1024_up4'
def main():


    train_set = get_set_patch(upscale_factor=upscale_factor, blur_kernel=(3, 3), noise_var=0.001, lesion=lesion, dataset=dataset, init_size=1024,
                                  mode='Train'
                                  , transform={'Crop_factor': Crop_factor, 'rotation_angle': 0})
    test_set = get_set_patch(upscale_factor=upscale_factor, blur_kernel=(3, 3), noise_var=0.001, lesion=lesion, dataset=dataset, init_size=1024,
                                 mode='Test', transform=None)
    training_data_loader = DataLoader(dataset=train_set, batch_size=1, num_workers=16,
                                          shuffle=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=16,
                                         shuffle=False)
    generator = Generator(upsample_factor=upscale_factor).cuda(device_ids[0])
    seg = U_Net_Cut(img_ch=3, output_ch=5).cuda(device_ids[0])

    generator = torch.nn.DataParallel(generator, device_ids)
    generator.load_state_dict(torch.load('pretrain_model/MICCAIGenerator/netG_epoch_4_50.pth',map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))
    if dataset=='DDR':
        seg.load_state_dict(torch.load('pretrain_model/DDR_seg_model_951.pth',map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))
    else:
        seg.load_state_dict(torch.load('pretrain_model/IDRiD_seg_model_651.pth', map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))
    seg = torch.nn.DataParallel(seg, device_ids)
    step=0

    img_bar = training_data_loader if mode=='Train' else testing_data_loader

    for step, (low_resolution, high_resolution_Y_linerhigh, high_resolution, true_masks) in enumerate(img_bar):
        step+=1

        low_resolution = low_resolution.cuda(device_ids[0])
        high_resolution = high_resolution.cuda(device_ids[0])

        true_masks= true_masks.numpy()[0]


        if np.max(true_masks[0])==1 or np.max(true_masks[1])==1 or np.max(true_masks[2])==1 or np.max(true_masks[3])==1:

            generator.eval()
            seg.eval()
            [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, sr] = generator(low_resolution)
            layer0 = layer0.cpu().detach().numpy()[0]
            layer1 = layer1.cpu().detach().numpy()[0]
            layer2 = layer2.cpu().detach().numpy()[0]
            layer3 = layer3.cpu().detach().numpy()[0]
            layer4 = layer4.cpu().detach().numpy()[0]
            layer5 = layer5.cpu().detach().numpy()[0]
            layer6 = layer6.cpu().detach().numpy()[0]
            layer7 = layer7.cpu().detach().numpy()[0]

            if seg_mode=='sr':
                seg_result=seg(sr)
                seg_result=seg_result.cpu().detach().numpy()[0]
            if seg_mode == 'hr':
                seg_result_hr = seg(high_resolution)
                seg_result_hr = seg_result_hr.cpu().detach().numpy()[0]


            sr = np.transpose(sr.cpu().detach().numpy()[0], (1, 2, 0))
            high_resolution = np.transpose(high_resolution.cpu().detach().numpy()[0], (1, 2, 0))

            # layer 0
            for i in range(layer0.shape[0]):
                if i==0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer0/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer0/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer0/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer0/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer0/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_4sr.jpg', sr)
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer0/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])

                img=layer0[i]
                max_img=np.max(img)
                img=np.clip(img,0,max_img)
                img=img/max_img

                plt.imsave('sr_vis/'+img_dir+'/layer0/img'+str(step)+'-layer0_feature'+str(i)+'.jpg',img)
                print('img',str(step),'  layer0','  feature',str(i))

            # layer 1
            for i in range(layer1.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer1/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer1[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/'+img_dir+'/layer1/img' + str(step) + '-layer0_feature' + str(i) + '.jpg', img)
                print('img', str(step), '  layer1', '  feature', str(i))

            # layer 2
            for i in range(layer2.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer2/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer2[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/'+img_dir+'/layer2/img' + str(step) + '-layer2_feature' + str(i) + '.jpg', img)
                print('img', str(step), '  layer2', '  feature', str(i))
            # layer 3
            for i in range(layer3.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer3/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer3[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/'+img_dir+'/layer3/img' + str(step) + '-layer3_feature' + str(i) + '.jpg', img)
                print('img', str(step), '  layer3', '  feature', str(i))
            # layer 4
            for i in range(layer4.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer4/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer4[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/'+img_dir+'/layer4/img' + str(step) + '-layer4_feature' + str(i) + '.jpg', img)
                print('img', str(step), '  layer4', '  feature', str(i))
            # layer 5
            for i in range(layer5.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer5/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer5[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/'+img_dir+'/layer5/img' + str(step) + '-layer5_feature' + str(i) + '.jpg', img)
                print('img', str(step), '  layer5', '  feature', str(i))

            # layer 6 deconv
            for i in range(layer6.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_4sr.jpg', sr)

                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer6[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/' + img_dir + '/layer6/img' + str(step) + '-layer6_feature' + str(i) + '.jpg',
                           img)
                print('img', str(step), '  layer6', '  feature', str(i))

            # layer 7 deconv
            for i in range(layer7.shape[0]):
                if i == 0:
                    if seg_mode == 'sr':
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '.jpg', high_resolution)
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_MA.jpg', true_masks[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HM.jpg', true_masks[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HE.jpg', true_masks[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_SE.jpg', true_masks[3])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_4sr.jpg', sr)
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_MA_srseg.jpg', seg_result[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HM_srseg.jpg', seg_result[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HE_srseg.jpg', seg_result[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_SE_srseg.jpg', seg_result[3])
                    if seg_mode == 'hr':
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_MA_hrseg.jpg', seg_result_hr[0])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HM_hrseg.jpg', seg_result_hr[1])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_HE_hrseg.jpg', seg_result_hr[2])
                        plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '_SE_hrseg.jpg', seg_result_hr[3])
                img = layer7[i]
                max_img = np.max(img)
                img = np.clip(img, 0, max_img)
                img = img / max_img

                plt.imsave('sr_vis/' + img_dir + '/layer7/img' + str(step) + '-layer7_feature' + str(i) + '.jpg',
                           img)
                print('img', str(step), '  layer7', '  feature', str(i))
            s=1




def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)



if __name__ == '__main__':
    init_seed = 103
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    import platform

    if seg_mode == 'sr':
        if not os.path.isdir('sr_vis/'+img_dir+'/layer0'):
            os.makedirs('sr_vis/'+img_dir+'/layer0')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer0')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer1'):
            os.makedirs('sr_vis/'+img_dir+'/layer1')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer1')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer2'):
            os.makedirs('sr_vis/'+img_dir+'/layer2')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer2')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer3'):
            os.makedirs('sr_vis/'+img_dir+'/layer3')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer3')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer4'):
            os.makedirs('sr_vis/'+img_dir+'/layer4')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer4')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer5'):
            os.makedirs('sr_vis/'+img_dir+'/layer5')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer5')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer6'):
            os.makedirs('sr_vis/'+img_dir+'/layer6')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer6')

        if not os.path.isdir('sr_vis/'+img_dir+'/layer7'):
            os.makedirs('sr_vis/'+img_dir+'/layer7')
        else:
            remove_all_file('sr_vis/'+img_dir+'/layer7')

    import warnings

    warnings.filterwarnings('ignore')
    main()


















