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
from network import MICCAI17_Generator as Generator
from network import U_Net_Cut, KeNet_v1, Multi_Intergrate
from util import LambdaLR
from torchvision.utils import save_image
from metrics import SSIM, computeAUPR
from torch.utils.data import DataLoader
from seg_loss import FocalLoss, DiceLoss
from data_process import get_set_patch_seg, display_transform, get_set_patch_cls
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
    lesion = {'MA': True, 'HM': True, 'HE': True, 'SE': True, 'BG': True}
    seg_CROSSENTROPY_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1]
    sr_loss_factor = {'mse': 1, 'tv': 1 * 1e-6, 'seg_perceptual': 1, 'cls_perceptual': 10,'cls':1}
    cls_loss_factor ={'ce':1,'seg_vis':0.3}
    num_thread = 16
    train_BATCH_SIZE = 1
    device_ids = [1]
    lr = {'cls_lr': 5 * 1e-5, 'seg_lr': 4 * 1e-6, 'g_lr': 1 * 1e-5 }
    upscale_factor = 8
    Crop_factor = 4
    seg_threshold = 0.5
    num_class = 5

blur_kernel = (3, 3)
noise_var = 0.001
Epoch = 300
num_epochs_decay = 150
test_BATCH_SIZE = 1

from sr_loss import PerceptionLoss, TVLoss, Seg_perceptual_loss

model_name = 'stage2' + '_upscale' + str(upscale_factor)


def main():
    num_lesion = 0
    for key, value in lesion.items():
        if value:
            num_lesion = num_lesion + 1
    train_set_cls = get_set_patch_cls(upscale_factor, blur_kernel, noise_var, dataset=dataset, init_size=1024,
                                      mode='Train')
    test_set_cls = get_set_patch_cls(upscale_factor, blur_kernel, noise_var, dataset=dataset, init_size=1024,
                                     mode='Test')
    training_data_loader_cls = DataLoader(dataset=train_set_cls, batch_size=train_BATCH_SIZE, num_workers=num_thread,
                                          shuffle=True)
    testing_data_loader_cls = DataLoader(dataset=test_set_cls, batch_size=test_BATCH_SIZE, num_workers=num_thread,
                                         shuffle=False)
    test_set_seg = get_set_patch_seg(upscale_factor, blur_kernel, noise_var, lesion, dataset=dataset, init_size=1024,
                                     mode='Test')
    testing_data_loader_seg = DataLoader(dataset=test_set_seg, batch_size=test_BATCH_SIZE, num_workers=num_thread,
                                         shuffle=False)
    print(model_name)

    generator = Generator(upsample_factor=upscale_factor).cuda(device_ids[0])
    generator = torch.nn.DataParallel(generator, device_ids)
    seg = U_Net_Cut(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
    seg = torch.nn.DataParallel(seg, device_ids)
    cls_net = KeNet_v1(classes_num=num_class).cuda(device_ids[0])
    cls_net = torch.nn.DataParallel(cls_net, device_ids)
    MultiScale_Intergrate = Multi_Intergrate().cuda(device_ids[0])
    MultiScale_Intergrate = torch.nn.DataParallel(MultiScale_Intergrate, device_ids)



    if dataset == 'DDR':
        if upscale_factor==8:
            generator.load_state_dict(torch.load('pretrain_model_stage3/MICCAIGenerator/generator_up8_180.pth',
                                           map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))
            seg.load_state_dict(torch.load('pretrain_model_stage3/Seg/DDR_seg_180_nosegStage2.pth',map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))
            cls_net.load_state_dict(torch.load('pretrain_model_stage3/Cls/DDR_cls_9_stage2_v3_new.pth',
                                           map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))
            MultiScale_Intergrate.load_state_dict(torch.load('pretrain_model_stage3/MVI/DDR_MultiScale_Intergrate_9.pth',
                                           map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))

    else:
        print('no idird pretrain seg for stage2')
        os._exit(0)


    ##################################
    ###    loss and optimizer      #
    ##################################

    optimizer_cls = optim.Adam(cls_net.parameters(), lr=lr['cls_lr'])
    optimizer_seg = optim.Adam(seg.parameters(), lr=lr['seg_lr'])
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr['g_lr'], betas=(0.9, 0.999))

    # cls loss
    if dataset == 'DDR':
        CE_criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([1, 10, 1.44, 25, 8])).float().cuda(device_ids[0]))
    else:
        CE_criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([1, 6.7, 1, 1.81, 2.73])).float().cuda(device_ids[0]))
    # sr loss
    MSE_criterion = nn.MSELoss().cuda(device_ids[0])
    seg_perceptual_criterion = nn.MSELoss().cuda(device_ids[0])
    cls_perceptual_criterion = nn.MSELoss().cuda(device_ids[0])
    seg_vis_criterion = nn.MSELoss().cuda(device_ids[0])
    TV_criterion = TVLoss().cuda(device_ids[0])

    # seg loss
    seg_criterion = nn.MSELoss().cuda(device_ids[0])


    writer = SummaryWriter(logdir='runs/runs_' + model_name + '_' + dataset)
    count = 0
    new_lr = lr
    results = {'cls_loss': [], 'seg_loss': [], 'g_loss': [],'segsr_loss':[],'clssr_loss':[], 'cls_acc': [], 'kappa': [],
               'psnr': [], 'ssim': [], 'PR_MA': [], 'PR_HM': [], 'PR_HE': [], 'PR_SE': [], 'AUC_MA': [], 'AUC_HM': [],
               'AUC_HE': [], 'AUC_SE': []}
    for epoch in range(0, Epoch):

        train_bar = tqdm(training_data_loader_cls)
        running_results = {'batch_sizes': 0, 'cls_loss': 0, 'seg_loss': 0, 'g_loss': 0}
        # Decay learning rate
        # if (epoch + 1) == num_epochs_decay:
        #     new_lr['g_lr'] = lr['g_lr'] / 5
        #     new_lr['cls_lr'] = lr['cls_lr'] / 2
        #     new_lr['seg_lr'] = lr['seg_lr'] / 2
        #     for param_group in optimizer_generator.param_groups:
        #         param_group['lr'] = new_lr['g_lr']
        #     for param_group in optimizer_seg.param_groups:
        #         param_group['lr'] = new_lr['vis_lr']
        #     for param_group in optimizer_cls.param_groups:
        #         param_group['lr'] = new_lr['cls_lr']

        """   train    """
        # remove_all_file('debug')
        if epoch >0:
            for low_resolution, high_resolution_Y_linerhigh, high_resolution, label in train_bar:

                count += 1
                batch_size = low_resolution.size(0)
                running_results['batch_sizes'] += batch_size
                generator.train()
                seg.train()
                cls_net.train()
                MultiScale_Intergrate.train()
                low_resolution = Variable(low_resolution).cuda(device_ids[0])
                high_resolution = Variable(high_resolution).cuda(device_ids[0])
                label = label.cuda(device_ids[0])

                ############################################
                #   training clsnet and vis parameters    #
                ############################################

                # obtain the segmentation map
                sr = generator(low_resolution)  # [N,3,1024,1024]
                masks_pred_sr = seg(sr)  # [N,5,1024,1024]
                masks_pred_sr = F.softmax(masks_pred_sr)
                Seperate_lesion_sr = masks_pred_sr[:, 0:4, :, :]  # [N,4,1024,1024]
                all_lesion_sr = torch.unsqueeze((1 - masks_pred_sr[:, 4, :, :]), 1)  # [N,1,1024,1024]
                cls_input_sr = torch.cat((sr, Seperate_lesion_sr), 1)  # [N,7,1024,1024]

                masks_pred_hr = seg(high_resolution)  # [N,5,1024,1024]
                masks_pred_hr = F.softmax(masks_pred_hr)
                Seperate_lesion_hr = masks_pred_hr[:, 0:4, :, :]  # [N,4,1024,1024]
                all_lesion_hr = torch.unsqueeze((1 - masks_pred_hr[:, 4, :, :]), 1)  # [N,1,1024,1024]
                cls_input_hr = torch.cat((high_resolution, Seperate_lesion_hr), 1)  # [N,7,1024,1024]

                # sr vis
                extractor = ModelOutputs(cls_net)
                features, fc_output = extractor(cls_input_sr)
                cam_all_sr = {'size256': [], 'size128': [], 'size64': [], 'size32': []}
                for batch in range(fc_output.shape[0]):
                    device = fc_output[batch].device.index
                    index = np.argmax(fc_output[batch].cpu().data.numpy())
                    if index == 0:
                        one_hot = np.zeros((1, fc_output[batch].size()[-1]), dtype=np.float32)
                        one_hot[0][0] = 1
                    if index > 0:
                        one_hot = np.ones((1, fc_output[batch].size()[-1]), dtype=np.float32)
                        one_hot[0][0] = 0
                    if batch == 0:
                        one_hot_all = Variable(torch.from_numpy(one_hot).cuda(device), requires_grad=True)
                    else:
                        one_hot_all = torch.cat(
                            (Variable(torch.from_numpy(one_hot).cuda(device), requires_grad=True), one_hot_all), 0)
                one_hot_all = torch.sum(one_hot_all * fc_output)
                cls_net.zero_grad()
                one_hot_all.backward(retain_graph=True)
                for i in range(4):
                    grads_val = extractor.get_gradients()[i][-1]
                    target = features[i][-1]
                    weights = torch.mean(grads_val, dim=(2, 3))
                    for batch in range(fc_output.shape[0]):
                        device = fc_output[batch].device.index
                        cam = torch.zeros(size=(target.shape[2], target.shape[3]), dtype=torch.float32).cuda(device)
                        for k, w in enumerate(weights[batch]):
                            cam += w * target[batch, k, :, :]
                        cam = nn.ReLU()(cam)
                        cam = cam - torch.min(cam)
                        if not torch.max(cam).cpu().detach().numpy() == 0:
                            cam = cam / torch.max(cam)
                        if i == 0:
                            cam_all_sr['size256'].append(cam)
                        elif i == 1:
                            cam_all_sr['size128'].append(cam)
                        elif i == 2:
                            cam_all_sr['size64'].append(cam)
                        elif i == 3:
                            cam_all_sr['size32'].append(cam)

                vis_sr = MultiScale_Intergrate(cam_all_sr)  # [N,1,1024,1024]
                cls_output_sr = fc_output

                # hr vis
                extractor = ModelOutputs(cls_net)
                features, fc_output = extractor(cls_input_hr)
                cam_all_hr = {'size256': [], 'size128': [], 'size64': [], 'size32': []}
                for batch in range(fc_output.shape[0]):
                    device = fc_output[batch].device.index
                    index = np.argmax(fc_output[batch].cpu().data.numpy())
                    if index == 0:
                        one_hot = np.zeros((1, fc_output[batch].size()[-1]), dtype=np.float32)
                        one_hot[0][0] = 1
                    if index > 0:
                        one_hot = np.ones((1, fc_output[batch].size()[-1]), dtype=np.float32)
                        one_hot[0][0] = 0
                    if batch == 0:
                        one_hot_all = Variable(torch.from_numpy(one_hot).cuda(device), requires_grad=True)
                    else:
                        one_hot_all = torch.cat(
                            (Variable(torch.from_numpy(one_hot).cuda(device), requires_grad=True), one_hot_all), 0)
                one_hot_all = torch.sum(one_hot_all * fc_output)
                cls_net.zero_grad()
                one_hot_all.backward(retain_graph=True)
                for i in range(4):
                    grads_val = extractor.get_gradients()[i][-1]
                    target = features[i][-1]
                    weights = torch.mean(grads_val, dim=(2, 3))
                    for batch in range(fc_output.shape[0]):
                        device = fc_output[batch].device.index
                        cam = torch.zeros(size=(target.shape[2], target.shape[3]), dtype=torch.float32).cuda(device)
                        for k, w in enumerate(weights[batch]):
                            cam += w * target[batch, k, :, :]
                        cam = nn.ReLU()(cam)
                        cam = cam - torch.min(cam)
                        if not torch.max(cam).cpu().detach().numpy() == 0:
                            cam = cam / torch.max(cam)
                        if i == 0:
                            cam_all_hr['size256'].append(cam)
                        elif i == 1:
                            cam_all_hr['size128'].append(cam)
                        elif i == 2:
                            cam_all_hr['size64'].append(cam)
                        elif i == 3:
                            cam_all_hr['size32'].append(cam)
                vis_hr = MultiScale_Intergrate(cam_all_hr)  # [N,1,1024,1024]


                #  seg perceptual loss
                seg_perceptual_loss = seg_perceptual_criterion(masks_pred_hr, masks_pred_sr)  # 0-1
                #  cls perceptual loss
                cls_perceptual_loss = cls_perceptual_criterion(vis_hr, vis_sr)  # 0-1
                #  seg-vis loss
                seg_vis_loss = seg_vis_criterion(vis_hr, masks_pred_hr)  # 0-1


                #  cls loss
                cls_loss = CE_criterion(cls_output_sr, label) * cls_loss_factor['ce']+seg_vis_loss* cls_loss_factor['seg_vis']

                #  sr loss
                MSE_loss = MSE_criterion(sr, high_resolution)
                tv_loss = TV_criterion(sr)
                generator_loss = MSE_loss * sr_loss_factor[
                    'mse'] + tv_loss * sr_loss_factor['tv'] + seg_perceptual_loss * sr_loss_factor['seg_perceptual'] \
                                 + cls_perceptual_loss * sr_loss_factor['cls_perceptual']+cls_loss * sr_loss_factor['cls']
                #  seg loss
                seg_loss=seg_vis_loss

                # update classification network
                optimizer_cls.zero_grad()
                cls_loss.backward(retain_graph=True)
                optimizer_cls.step()

                # update seg net
                # optimizer_seg.zero_grad()
                # seg_loss.backward(retain_graph=True)
                # optimizer_seg.step()

                # update sr net
                optimizer_generator.zero_grad()
                generator_loss.backward()
                optimizer_generator.step()



                # loss for current batch before optimization
                running_results['cls_loss'] += cls_loss.item() * batch_size
                running_results['seg_loss'] += seg_loss.item() * batch_size
                running_results['g_loss'] += generator_loss.item() * batch_size


                train_bar.set_description(
                    desc='[%d/%d] cls_loss: %.4f seg_loss: %.4f sr_loss: %.4f' % (
                        epoch, Epoch,
                        running_results['cls_loss'] / running_results['batch_sizes'],
                        running_results['seg_loss'] / running_results['batch_sizes'],
                        running_results['g_loss'] / running_results['batch_sizes'],

                    ))

        """------------------Test--------------"""
        if epoch % 1 == 0:
            with torch.no_grad():
                count_val_cls = 0
                count_val_seg = 0
                SR_val_images = []
                Seg_val_images = {'MA': [], 'HM': [], 'HE': [], 'SE': []}
                val_bar_cls = tqdm(testing_data_loader_cls)
                val_bar_seg = tqdm(testing_data_loader_seg)
                valing_results = {'batch_sizes': 0, 'cls_acc': 0, 'kappa': 0,
                                  'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0,  'batch_sizes_MA': 0,
                                  'batch_sizes_HM': 0, 'batch_sizes_HE': 0, 'batch_sizes_SE': 0, 'AUC_MA': 0,
                                  'AUC_HM': 0, 'AUC_HE': 0, 'AUC_SE': 0, 'PR_MA': 0, 'PR_HM': 0, 'PR_HE': 0,
                                  'PR_SE': 0
                                  }
                correct_all = 0
                total_all = 0
                correct_perclass = [0, 0, 0, 0, 0]
                total_perclass = [0, 0, 0, 0, 0]
                Seg_valing_results = {'AUC_MA': 0, 'AUC_HM': 0, 'AUC_HE': 0, 'AUC_SE': 0, 'PR_MA': 0, 'PR_HM': 0,
                                      'PR_HE': 0, 'PR_SE': 0}

                """ test segmentation """
                print("segmentation Test!")
                for low_resolution, high_resolution_linerhigh, high_resolution, true_masks in val_bar_seg:
                    batch_size = low_resolution.size(0)
                    valing_results['batch_sizes'] += batch_size
                    generator.eval()
                    seg.eval()
                    high_resolution_linerhigh = Variable(high_resolution_linerhigh).cuda(device_ids[0])
                    low_resolution = Variable(low_resolution).cuda(device_ids[0])
                    high_resolution = Variable(high_resolution).cuda(device_ids[0])
                    true_masks = Variable(true_masks).cuda(device_ids[0])
                    GT = true_masks.cpu().numpy()[0]  # [5,1024,1024]

                    # plt.imsave('img_result/img.jpg',GT[2])

                    MA_GT = GT[0].reshape(-1).astype(int)
                    HM_GT = GT[1].reshape(-1).astype(int)  # [1024*1024]
                    HE_GT = GT[2].reshape(-1).astype(int)
                    SE_GT = GT[3].reshape(-1).astype(int)
                    # SR metrics calculate
                    sr = generator(low_resolution)
                    batch_mse = ((sr - high_resolution) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, high_resolution).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                    # Seg metrics calculate
                    masks_pred = seg(sr)
                    masks_pred = torch.sigmoid(masks_pred)  # [1,5,1024,1024]
                    result = masks_pred.cpu().numpy()[0]  # [5,1024,1024]
                    MA_pred_scores = result[0].reshape(-1)
                    HM_pred_scores = result[1].reshape(-1)
                    HE_pred_scores = result[2].reshape(-1)
                    SE_pred_scores = result[3].reshape(-1)

                    if np.sum(MA_GT) != 0:
                        valing_results['AUC_MA'] += roc_auc_score(y_true=MA_GT, y_score=MA_pred_scores)
                        valing_results['PR_MA'] += average_precision_score(y_true=MA_GT, y_score=MA_pred_scores)
                        valing_results['batch_sizes_MA'] += batch_size
                        Seg_valing_results['AUC_MA'] = valing_results['AUC_MA'] / valing_results['batch_sizes_MA']
                        Seg_valing_results['PR_MA'] = valing_results['PR_MA'] / valing_results['batch_sizes_MA']
                    if np.sum(HM_GT) != 0:
                        valing_results['AUC_HM'] += roc_auc_score(y_true=HM_GT, y_score=HM_pred_scores)
                        valing_results['PR_HM'] += average_precision_score(y_true=HM_GT, y_score=HM_pred_scores)
                        valing_results['batch_sizes_HM'] += batch_size
                        Seg_valing_results['AUC_HM'] = valing_results['AUC_HM'] / valing_results['batch_sizes_HM']
                        Seg_valing_results['PR_HM'] = valing_results['PR_HM'] / valing_results['batch_sizes_HM']
                    if np.sum(HE_GT) != 0:
                        valing_results['AUC_HE'] += roc_auc_score(y_true=HE_GT, y_score=HE_pred_scores)
                        valing_results['PR_HE'] += average_precision_score(y_true=HE_GT, y_score=HE_pred_scores)
                        valing_results['batch_sizes_HE'] += batch_size
                        Seg_valing_results['AUC_HE'] = valing_results['AUC_HE'] / valing_results['batch_sizes_HE']
                        Seg_valing_results['PR_HE'] = valing_results['PR_HE'] / valing_results['batch_sizes_HE']
                    if np.sum(SE_GT) != 0:
                        valing_results['AUC_SE'] += roc_auc_score(y_true=SE_GT, y_score=SE_pred_scores)
                        valing_results['PR_SE'] += average_precision_score(y_true=SE_GT, y_score=SE_pred_scores)
                        valing_results['batch_sizes_SE'] += batch_size
                        Seg_valing_results['AUC_SE'] = valing_results['AUC_SE'] / valing_results['batch_sizes_SE']
                        Seg_valing_results['PR_SE'] = valing_results['PR_SE'] / valing_results['batch_sizes_SE']
                    val_bar_seg.set_description(
                        desc='[Test!] PSNR: %.4f dB SSIM: %.4f |MA_PR: %.4f |HM_PR: %.4f |HE_PR: %.4f |SE_PR: %.4f'
                             % (valing_results['psnr'], valing_results['ssim'], Seg_valing_results['PR_MA'],
                                Seg_valing_results['PR_HM']
                                , Seg_valing_results['PR_HE'], Seg_valing_results['PR_SE']))
                    GT = GT.astype(np.float32)
                    high_resolution_linerhigh=high_resolution_linerhigh.cpu()
                    if count_val_seg == 0 or count_val_seg == 1 or count_val_seg == 2 or count_val_seg == 3 or count_val_seg == 4:
                        SR_val_images.extend(
                            [display_transform()(high_resolution_linerhigh.squeeze(0)),
                             display_transform()(high_resolution.data.cpu().squeeze(0)),
                             display_transform()(sr.data.cpu().squeeze(0))])

                        Seg_val_images['MA'].extend(
                            [display_transform()(high_resolution.data.cpu().squeeze(0)),
                             display_transform()(torch.from_numpy(tile(np.expand_dims(GT[0], axis=0), (3, 1, 1)))),
                             display_transform()(
                                 torch.from_numpy(tile(np.expand_dims(result[0], axis=0), (3, 1, 1))))])
                        Seg_val_images['HM'].extend(
                            [display_transform()(high_resolution.data.cpu().squeeze(0)),
                             display_transform()(torch.from_numpy(tile(np.expand_dims(GT[1], axis=0), (3, 1, 1)))),
                             display_transform()(
                                 torch.from_numpy(tile(np.expand_dims(result[1], axis=0), (3, 1, 1))))])
                        Seg_val_images['HE'].extend(
                            [display_transform()(high_resolution.data.cpu().squeeze(0)),
                             display_transform()(torch.from_numpy(tile(np.expand_dims(GT[2], axis=0), (3, 1, 1)))),
                             display_transform()(
                                 torch.from_numpy(tile(np.expand_dims(result[2], axis=0), (3, 1, 1))))])
                        Seg_val_images['SE'].extend(
                            [display_transform()(high_resolution.data.cpu().squeeze(0)),
                             display_transform()(torch.from_numpy(tile(np.expand_dims(GT[3], axis=0), (3, 1, 1)))),
                             display_transform()(
                                 torch.from_numpy(tile(np.expand_dims(result[3], axis=0), (3, 1, 1))))])

                    count_val_seg += 1
                """ test Classification """
                print("Classification Test!")
                for low_resolution, high_resolution_linerhigh, high_resolution, label in val_bar_cls:
                    generator.eval()
                    seg.eval()
                    cls_net.eval()
                    low_resolution = Variable(low_resolution).cuda(device_ids[0])
                    label = label.cuda(device_ids[0])

                    # SR metrics calculate
                    sr=generator(low_resolution)
                    masks_pred = seg(sr)  # [N,5,1024,1024]
                    masks_pred = F.softmax(masks_pred)
                    Seperate_lesion = masks_pred[:, 0:4, :, :]  # [N,4,1024,1024]
                    all_lesion = torch.unsqueeze((1 - masks_pred[:, 4, :, :]), 1)  # [N,1,1024,1024]
                    cls_input = torch.cat((sr, Seperate_lesion), 1)  # [N,7,1024,1024]

                    cls_output_sr = cls_net(cls_input)
                    _, predicted = torch.max(cls_output_sr.data, 1)
                    total_all += label.size(0)
                    correct_all += (predicted == label).sum()
                    label = label.cpu().numpy()
                    predicted = predicted.cpu().numpy()
                    for i_test in range(test_BATCH_SIZE):
                        total_perclass[label[i_test]] += 1
                        if predicted[i_test] == label[i_test]:
                            correct_perclass[label[i_test]] += 1
                    valing_results['cls_acc'] = 100. * correct_all.cpu().numpy() / total_all
                    p_o = correct_all.cpu().numpy() / total_all
                    p_e = (correct_perclass[0] * total_perclass[0] + correct_perclass[1] * total_perclass[1] +
                           correct_perclass[2] * total_perclass[2]
                           + correct_perclass[3] * total_perclass[3] + correct_perclass[4] * total_perclass[4]) / (
                                  total_all * total_all)
                    valing_results['kappa'] = (p_o - p_e) / (1 - p_e)
                    val_bar_cls.set_description(
                        desc='[Test!]OA: %.4f |kappa: %.4f '
                             % (valing_results['cls_acc'], valing_results['kappa']))
                    count_val_cls += 1


                SR_val_images = torch.stack(SR_val_images)
                SR_val_images = torch.chunk(SR_val_images, SR_val_images.size(0) // 15)
                Seg_val_images_MA = torch.stack(Seg_val_images['MA'])
                Seg_val_images_MA = torch.chunk(Seg_val_images_MA, Seg_val_images_MA.size(0) // 15)
                Seg_val_images_HM = torch.stack(Seg_val_images['HM'])
                Seg_val_images_HM = torch.chunk(Seg_val_images_HM, Seg_val_images_HM.size(0) // 15)
                Seg_val_images_HE = torch.stack(Seg_val_images['HE'])
                Seg_val_images_HE = torch.chunk(Seg_val_images_HE, Seg_val_images_HE.size(0) // 15)
                Seg_val_images_SE = torch.stack(Seg_val_images['SE'])
                Seg_val_images_SE = torch.chunk(Seg_val_images_SE, Seg_val_images_SE.size(0) // 15)
                SR_val_save_bar = tqdm(SR_val_images, desc='[saving SR results]')
                Seg_MA_val_save_bar = tqdm(Seg_val_images_MA, desc='[saving Seg_MA results]')
                Seg_HM_val_save_bar = tqdm(Seg_val_images_HM, desc='[saving Seg_HM results]')
                Seg_HE_val_save_bar = tqdm(Seg_val_images_HE, desc='[saving Seg_HE results]')
                Seg_SE_val_save_bar = tqdm(Seg_val_images_SE, desc='[saving Seg_SE results]')
                index = 1
                for image in SR_val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image,
                                     'img_result_sr/' + model_name + '_' + dataset + '/' + 'epoch_%d_index_%d.png' % (
                                         epoch, index), padding=5)
                for image in Seg_MA_val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image,
                                     'img_result_seg/' + model_name + '_' + dataset + '/' + 'epoch_%d_index_%d_MA.png' % (
                                         epoch, index), padding=5)
                for image in Seg_HM_val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image,
                                     'img_result_seg/' + model_name + '_' + dataset + '/' + 'epoch_%d_index_%d_HM.png' % (
                                         epoch, index), padding=5)
                for image in Seg_HE_val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image,
                                     'img_result_seg/' + model_name + '_' + dataset + '/' + 'epoch_%d_index_%d_HE.png' % (
                                         epoch, index), padding=5)
                for image in Seg_SE_val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image,
                                     'img_result_seg/' + model_name + '_' + dataset + '/' + 'epoch_%d_index_%d_SE.png' % (
                                         epoch, index), padding=5)


            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])
            results['PR_MA'].append(Seg_valing_results['PR_MA'])
            results['PR_HM'].append(Seg_valing_results['PR_HM'])
            results['PR_HE'].append(Seg_valing_results['PR_HE'])
            results['PR_SE'].append(Seg_valing_results['PR_SE'])
            results['AUC_MA'].append(Seg_valing_results['AUC_MA'])
            results['AUC_HM'].append(Seg_valing_results['AUC_HM'])
            results['AUC_HE'].append(Seg_valing_results['AUC_HE'])
            results['AUC_SE'].append(Seg_valing_results['AUC_SE'])
            results['cls_acc'].append(valing_results['cls_acc'])
            results['kappa'].append(valing_results['kappa'])

            out_path = 'statistics/' + model_name + '_' + dataset + '/'
            data_frame = pd.DataFrame(
                data={
                      'PSNR': results['psnr'], 'SSIM': results['ssim'], 'PR_MA': results['PR_MA'],
                      'PR_HM': results['PR_HM'], 'PR_HE': results['PR_HE'], 'PR_SE': results['PR_SE'],
                      'AUC_MA': results['AUC_MA'], 'AUC_HM': results['AUC_HM'], 'AUC_HE': results['AUC_HE'],
                      'AUC_SE': results['AUC_SE'],
                'cls_acc': results['cls_acc'], 'kappa': results['kappa']},
            )
            data_frame.to_csv(out_path + model_name + '_train_results.csv',
                              index_label='Epoch')

            torch.save(cls_net.state_dict(),
                       'model/model_' + model_name + '_' + dataset + '/cls_' + str(epoch + 1) + '.pth')
            torch.save(MultiScale_Intergrate.state_dict(),
                       'model/model_' + model_name + '_' + dataset + '/MultiScale_Intergrate' + str(epoch + 1) + '.pth')
            torch.save(seg.state_dict(),
                       'model/model_' + model_name + '_' + dataset + '/seg_' + str(epoch + 1) + '.pth')
            torch.save(generator.state_dict(),
                       'model/model_' + model_name + '_' + dataset + '/generator' + str(epoch + 1) + '.pth')


        writer.close()


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


class FeatureExtractor():
    def __init__(self, model):
        self.model = model

        self.gradients_256 = []
        self.gradients_128 = []
        self.gradients_64 = []
        self.gradients_32 = []
        self.dense_spp = nn.Sequential(
            nn.Linear(10752, 512),
            nn.Dropout(0.5),
            nn.Linear(512, classes_num)
        )

    def save_gradient_256(self, grad):
        self.gradients_256.append(grad)

    def save_gradient_128(self, grad):
        self.gradients_128.append(grad)

    def save_gradient_64(self, grad):
        self.gradients_64.append(grad)

    def save_gradient_32(self, grad):
        self.gradients_32.append(grad)

    def __call__(self, x):  # x~[B,3,128,128]
        target_activations = [[], [], [], []]

        x = self.model._modules['module']._modules['new_first_layer'](x)
        x = self.model._modules['module']._modules['features']._modules['1'](x)
        x = self.model._modules['module']._modules['features']._modules['2'](x)
        x = self.model._modules['module']._modules['features']._modules['3'](x)  # 256*256
        x = self.model._modules['module']._modules['features']._modules['4'](x)  # 256*256
        x.register_hook(self.save_gradient_256)
        target_activations[0] += [x]
        x = self.model._modules['module']._modules['features']._modules['5'](x)  # 128*128
        x.register_hook(self.save_gradient_128)
        target_activations[1] += [x]
        x = self.model._modules['module']._modules['features']._modules['6'](x)  # 64*64
        x.register_hook(self.save_gradient_64)
        target_activations[2] += [x]
        x = self.model._modules['module']._modules['features']._modules['7'](x)  # 32*32
        x.register_hook(self.save_gradient_32)
        target_activations[3] += [x]
        # x = self.model._modules['module']._modules['features'](x)

        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(x.size(0), -1)
        # x_classifier = self.model._modules['module']._modules['denses'](x)

        spp = spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)
        x_classifier = self.dense_spp(spp)

        return target_activations, x_classifier


class ModelOutputs():

    def __init__(self, model):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model)

    def get_gradients(self):
        return [self.feature_extractor.gradients_256, self.feature_extractor.gradients_128,
                self.feature_extractor.gradients_64, self.feature_extractor.gradients_32]

    def __call__(self, x):
        target_activations, fc_output = self.feature_extractor(x)

        return target_activations, fc_output


if __name__ == '__main__':
    init_seed = 103
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    import platform

    sysstr = platform.system()
    if (sysstr == "Linux"):
        if not os.path.isdir('model/model_' + model_name + '_' + dataset):
            os.makedirs('model/model_' + model_name + '_' + dataset)
        else:
            remove_all_file('model/model_' + model_name + '_' + dataset)

        if not os.path.isdir('img_result_sr/' + model_name + '_' + dataset):
            os.makedirs('img_result_sr/' + model_name + '_' + dataset)
        else:
            remove_all_file('img_result_sr/' + model_name + '_' + dataset)
        if not os.path.isdir('statistics/' + model_name + '_' + dataset):
            os.makedirs('statistics/' + model_name + '_' + dataset)
        else:
            remove_all_file('statistics/' + model_name + '_' + dataset)

        if not os.path.isdir('img_result_seg/' + model_name + '_' + dataset):
            os.makedirs('img_result_seg/' + model_name + '_' + dataset)
        else:
            remove_all_file('img_result_seg/' + model_name + '_' + dataset)

        if os.path.isdir('runs/runs_' + model_name + '_' + dataset):
            remove_all_file('runs/runs_' + model_name + '_' + dataset)
    import warnings

    warnings.filterwarnings('ignore')
    main()
