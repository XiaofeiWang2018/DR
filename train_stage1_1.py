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
from network import U_Net_Cut
from util import LambdaLR
from torchvision.utils import save_image
from metrics import SSIM,computeAUPR
from torch.utils.data import DataLoader
from seg_loss import FocalLoss, DiceLoss
from data_process import get_set_patch_seg,display_transform
from math import log10
from matplotlib import pyplot as plt
import scipy.misc
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_ssim
import pandas as pd
"""
sr(pre)+seg(pre)
"""
if 1:
    dataset = 'DDR'
    lesion = {'MA': True, 'HM': True, 'HE': True, 'SE': True, 'BG': True}
    seg_CROSSENTROPY_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1]
    sr_loss_factor = {'mse': 1, 'tv': 1 * 1e-6,'Seg_perceptual':0.4,'seg':1}
    seg_loss_factor = {'Dice': 1, 'Focal': 1}
    num_thread = 16
    train_BATCH_SIZE = 2
    device_ids = [1]
    lr = {'g_lr': 5 * 1e-5,  'seg_lr': 2 * 1e-5}
    upscale_factor = 8
    Crop_factor = 4
    seg_threshold = 0.5
    num_class=5
    pre_train=False


blur_kernel = (3, 3)
noise_var = 0.001
Epoch = 30000
num_epochs_decay=150
test_BATCH_SIZE = 1
from sr_loss import PerceptionLoss, TVLoss, Seg_perceptual_loss
model_name = 'stage1_1'+ '_upscale' + str(upscale_factor)

def main():
    num_lesion = 0
    for key, value in lesion.items():
        if value:
            num_lesion = num_lesion + 1
    train_set = get_set_patch_seg(upscale_factor, blur_kernel, noise_var, lesion, dataset=dataset, init_size=1024,
                              mode='Train'
                              , transform={'Crop_factor': Crop_factor, 'rotation_angle': 0})
    test_set = get_set_patch_seg(upscale_factor, blur_kernel, noise_var, lesion, dataset=dataset, init_size=1024,
                             mode='Test', transform=None)
    training_data_loader = DataLoader(dataset=train_set, batch_size=train_BATCH_SIZE, num_workers=num_thread,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=test_BATCH_SIZE, num_workers=num_thread,
                                     shuffle=False)
    print(model_name)
    # network and initialization
    generator = ISR_Net(upsample_factor=upscale_factor).cuda(device_ids[0])
    seg = Seg_Net(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
    seg = torch.nn.DataParallel(seg, device_ids)
    generator = torch.nn.DataParallel(generator, device_ids)
    if pre_train:
        if upscale_factor == 4:
            generator.load_state_dict(torch.load('pre_train isr_up4',map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))
        elif upscale_factor == 8:
            generator.load_state_dict(torch.load('pre_train isr_up8',
                                                 map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))
        if dataset=='DDR':
            seg.load_state_dict(torch.load('pre_train seg',map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))



    ##################################
    ###    loss and optimizer      #
    ##################################

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr['g_lr'], betas=(0.9, 0.999))
    optimizer_seg = optim.Adam(seg.parameters(), lr=lr['seg_lr'], betas=(0.9, 0.999))

    # sr loss
    MSE_criterion = nn.MSELoss().cuda(device_ids[0])
    Seg_perceptual_criterion = nn.MSELoss().cuda(device_ids[0])
    TV_criterion = TVLoss().cuda(device_ids[0])
    # seg loss
    Dice_criterion = DiceLoss(weight=torch.FloatTensor(seg_CROSSENTROPY_WEIGHTS).cuda(device_ids[0]),device_ids=device_ids)
    Focal_criterion = FocalLoss().cuda(device_ids[0])

    writer = SummaryWriter(logdir='runs/runs_' + model_name + '_' + dataset)
    count = 0
    new_lr = lr
    results = {'d_loss': [], 'g_loss': [], 'Dice_loss': [],'Focal_loss': [],'SegSR_loss':[],
               'psnr': [], 'ssim': [],'PR_MA': [],'PR_HM': [],'PR_HE': [],'PR_SE': [],'AUC_MA': [],'AUC_HM': [],'AUC_HE': [],'AUC_SE': [],
                }
    with open('metrics/metrics_' + model_name + '_' + dataset + '.txt', "w+") as f:
        for epoch in range(0, Epoch):

            train_bar = tqdm(training_data_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0,'mse_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0,'Seg_loss':0,'Focal_loss':0,'Dice_loss':0,'SegSR_loss':0}
            # Decay learning rate
            if (epoch + 1) == num_epochs_decay:
                new_lr['g_lr'] = lr['g_lr']/5
                new_lr['seg_lr'] = lr['seg_lr']/2
                for param_group in optimizer_generator.param_groups:
                    param_group['lr'] = new_lr['g_lr']
                for param_group in optimizer_seg.param_groups:
                    param_group['lr'] = new_lr['seg_lr']
                print('Decay learning rate to lr: {}.'.format(new_lr))

            """   train    """
            for low_resolution, high_resolution_Y_linerhigh, high_resolution, true_masks in train_bar:

                count += 1
                batch_size = low_resolution.size(0)
                running_results['batch_sizes'] += batch_size
                generator.train()
                seg.train()

                low_resolution = Variable(low_resolution).cuda(device_ids[0])
                high_resolution = Variable(high_resolution).cuda(device_ids[0])
                high_resolution_Y_linerhigh = Variable(high_resolution_Y_linerhigh).cuda(device_ids[0])
                true_masks = Variable(true_masks).cuda(device_ids[0])# [1,5,256,256]

                ############################################
                #   training generator and segmentation    #
                ############################################
                fake_high_resolution = generator(low_resolution)
                optimizer_seg.zero_grad()
                optimizer_generator.zero_grad()
                true_masks_transpose = true_masks.permute(0, 2, 3, 1)  # [1,256,256,5]
                true_masks_pred_flat = true_masks_transpose.reshape(-1, true_masks_transpose.shape[-1])  # [65536,5]
                masks_pred = seg(fake_high_resolution)  # [1,5,256,256]
                masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)  # [1,256,256,5]
                masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])  # [65536,5]
                true_masks_indices = torch.argmax(true_masks, 1)  # [1,256,256]

                # hr seg map
                masks_pred_hr = seg(high_resolution)  # [1,5,256,256]
                masks_pred_transpose_hr = masks_pred_hr.permute(0, 2, 3, 1)  # [1,256,256,5]
                masks_pred_flat_hr = masks_pred_transpose_hr.reshape(-1,masks_pred_transpose_hr.shape[-1])  # [65536,5]

                # sr loss
                MSE_loss = MSE_criterion(fake_high_resolution, high_resolution)
                tv_loss = TV_criterion(fake_high_resolution)

                # seg loss
                Dice_loss = Dice_criterion(masks_pred_flat, true_masks_pred_flat,transform='sigmoid',device_ids=device_ids)
                Focal_loss = Focal_criterion(masks_pred, true_masks_indices,transform='sigmoid')
                # seg perceptual loss
                Seg_perceptual_loss = Seg_perceptual_criterion(masks_pred_flat, masks_pred_flat_hr)  # 0-1
                # # seg network update
                seg_loss = Dice_loss * seg_loss_factor['Dice'] + Focal_loss * seg_loss_factor['Focal']
                seg_loss.backward(retain_graph=True)
                optimizer_seg.step()
                # sr network update
                generator_loss = MSE_loss * sr_loss_factor[
                    'mse'] + tv_loss * sr_loss_factor['tv']+Seg_perceptual_loss* sr_loss_factor['Seg_perceptual']+seg_loss * sr_loss_factor['seg']
                generator_loss.backward()
                optimizer_generator.step()

                # loss for current batch before optimization
                running_results['g_loss'] += generator_loss.item() * batch_size
                running_results['mse_loss'] += MSE_loss.item() * batch_size
                running_results['Seg_loss'] += seg_loss.item() * batch_size
                running_results['Focal_loss'] += Focal_loss.item() * batch_size
                running_results['Dice_loss'] += Dice_loss.item() * batch_size
                running_results['SegSR_loss'] += Seg_perceptual_loss.item() * batch_size

                train_bar.set_description(desc='[%d/%d] mse_loss: %.4f SegSR_loss: %.4f Focal_loss: %.4f Dice_loss: %.4f' % (
                    epoch, Epoch,
                    running_results['mse_loss'] / running_results['batch_sizes'],
                    running_results['SegSR_loss'] / running_results['batch_sizes'],
                    running_results['Focal_loss'] / running_results['batch_sizes'],
                    running_results['Dice_loss'] / running_results['batch_sizes'],
                ))

                """------------------tensorboard TRAIN--------------"""
                if count % 100 == 0:
                    writer.add_scalar('scalar/SR_G_loss', generator_loss.item(), count)
                    writer.add_scalar('scalar/Seg_loss', seg_loss.item(), count)
                    writer.add_scalar('scalar/SR_content_loss', MSE_loss.item() , count)
                    writer.add_scalar('scalar/SR_tv_loss', tv_loss.item(), count)
                    writer.add_scalar('scalar/SR_Seg_perceptual_loss', Seg_perceptual_loss.item(), count)
                    writer.add_scalar('scalar/Seg_Focal_loss', Focal_loss.item() ,count)
                    writer.add_scalar('scalar/Seg_Dice_loss', Dice_loss.item() , count)

            """------------------Test--------------"""
            if epoch % 4 == 0 :
                print("Waiting Test!")
                with torch.no_grad():
                    count_val = 0
                    SR_val_images = []
                    Seg_val_images = {'MA': [], 'HM': [], 'HE': [], 'SE': []}
                    val_bar = tqdm(testing_data_loader)
                    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'batch_sizes_MA': 0,
                                      'batch_sizes_HM': 0, 'batch_sizes_HE': 0, 'batch_sizes_SE': 0, 'AUC_MA': 0,
                                      'AUC_HM': 0, 'AUC_HE': 0, 'AUC_SE': 0, 'PR_MA': 0, 'PR_HM': 0, 'PR_HE': 0,
                                      'PR_SE': 0}
                    Seg_valing_results = {'AUC_MA': 0, 'AUC_HM': 0, 'AUC_HE': 0, 'AUC_SE': 0, 'PR_MA': 0, 'PR_HM': 0,
                                          'PR_HE': 0, 'PR_SE': 0}
                    for low_resolution, high_resolution_linerhigh, high_resolution, true_masks in val_bar:
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
                        val_bar.set_description(
                            desc='[Test!] PSNR: %.4f dB SSIM: %.4f |MA_PR: %.4f |HM_PR: %.4f |HE_PR: %.4f |SE_PR: %.4f'
                                 % (valing_results['psnr'], valing_results['ssim'], Seg_valing_results['PR_MA'],
                                    Seg_valing_results['PR_HM']
                                    , Seg_valing_results['PR_HE'], Seg_valing_results['PR_SE']))
                        GT = GT.astype(np.float32)
                        high_resolution_linerhigh=high_resolution_linerhigh.cpu()
                        if count_val == 0 or count_val == 1 or count_val == 2 or count_val == 3 or count_val == 4:
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

                        count_val += 1

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

                torch.save(generator.state_dict(),
                           'model/model_' + model_name + '_' + dataset + '/generator_' + str(epoch) + '.pth')
                torch.save(seg.state_dict(),
                           'model/model_' + model_name + '_' + dataset + '/seg_' + str(epoch ) + '.pth')
                # save loss\scores\psnr\ssim
                if epoch != 0:
                    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
                    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
                    results['Dice_loss'].append(running_results['Dice_loss'] / running_results['batch_sizes'])
                    results['Focal_loss'].append(running_results['Focal_loss'] / running_results['batch_sizes'])
                    results['SegSR_loss'].append(running_results['SegSR_loss'] / running_results['batch_sizes'])
                else:
                    results['d_loss'].append(0)
                    results['g_loss'].append(0)
                    results['Dice_loss'].append(0)
                    results['Focal_loss'].append(0)
                    results['SegSR_loss'].append(0)

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



                out_path = 'statistics/' + model_name + '_' + dataset + '/'
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                          'Dice_loss': results['Dice_loss'], 'Focal_loss': results['Focal_loss'],
                          'SegSR_loss': results['SegSR_loss'],
                          'PSNR': results['psnr'], 'SSIM': results['ssim'], 'PR_MA': results['PR_MA'],
                          'PR_HM': results['PR_HM'], 'PR_HE': results['PR_HE'], 'PR_SE': results['PR_SE'],
                          'AUC_MA': results['AUC_MA'], 'AUC_HM': results['AUC_HM'], 'AUC_HE': results['AUC_HE'],
                          'AUC_SE': results['AUC_SE']},
                )
                data_frame.to_csv(out_path + 'srf_' + str(upscale_factor) + '_train_results.csv',
                                  index_label='Epoch')




        writer.close()


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


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
