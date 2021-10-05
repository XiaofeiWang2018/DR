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
from network import U_Net_Cut,KeNet_v1,Multi_Intergrate
from util import LambdaLR
from torchvision.utils import save_image
from metrics import SSIM,computeAUPR
from torch.utils.data import DataLoader
from seg_loss import FocalLoss, DiceLoss
from data_process import get_set_patch_seg,display_transform,get_set_patch_cls
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
    num_thread = 8
    train_BATCH_SIZE = 2
    device_ids = [1]
    lr_change_epoch=10
    lr = {'cls_lr': 5*1e-5,'vis_lr': 1*1e-5,'seg_lr': 4*1e-6}
    upscale_factor = 8
    Crop_factor = 4
    seg_threshold = 0.5
    num_class=5


blur_kernel = (3, 3)
noise_var = 0.001
Epoch = 300
test_BATCH_SIZE = 1

from sr_loss import PerceptionLoss, TVLoss, Seg_perceptual_loss
model_name = 'stage1_2'+ '_upscale' + str(upscale_factor)

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
    print(model_name)


    seg = U_Net_Cut(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
    seg = torch.nn.DataParallel(seg, device_ids)
    cls_net = KeNet_v1(classes_num=num_class).cuda(device_ids[0])
    cls_net = torch.nn.DataParallel(cls_net, device_ids)
    MultiScale_Intergrate = Multi_Intergrate().cuda(device_ids[0])
    MultiScale_Intergrate = torch.nn.DataParallel(MultiScale_Intergrate, device_ids)

    if dataset=='DDR':
        seg.load_state_dict(torch.load('pretrain_model_stage2/Seg/DDR_seg_180.pth',map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))
    else:
        print('no idird pretrain seg for stage2')
        os._exit(0)



    ##################################
    ###    loss and optimizer      #
    ##################################

    optimizer_cls = optim.Adam(cls_net.parameters(), lr=lr['cls_lr'])
    optimizer_seg = optim.Adam(seg.parameters(), lr=lr['seg_lr'])
    optimizer_MVI = optim.Adam(MultiScale_Intergrate.parameters(), lr=lr['vis_lr'])


    # cls loss
    if dataset == 'DDR':
        CE_criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([1, 10, 1.44, 25, 8])).float().cuda(device_ids[0]))
    else:
        CE_criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([1, 6.7, 1, 1.81, 2.73])).float().cuda(device_ids[0]))
    # visualization loss
    MSE_criterion = nn.MSELoss().cuda(device_ids[0])

    writer = SummaryWriter(logdir='runs/runs_' + model_name + '_' + dataset)
    count = 0
    new_lr_cls = lr['cls_lr']
    new_lr_vis = lr['vis_lr']
    results = {'cls_loss':[],'MVI_loss':[],'seg_loss': [],'cls_acc': [], 'kappa': []}
    with open('metrics/metrics_' + model_name + '_' + dataset + '.txt', "w+") as f:
        for epoch in range(0, Epoch):
            if (epoch + 1) == lr_change_epoch:
                new_lr_cls = new_lr_cls/5
                new_lr_vis = new_lr_vis *5
                for param_group in optimizer_cls.param_groups:
                    param_group['lr'] = new_lr_cls
                for param_group in optimizer_MVI.param_groups:
                    param_group['lr'] = new_lr_vis
                print('Decay cls learning rate to lr: {}.'.format(new_lr_cls))
                print('Decay vis learning rate to lr: {}.'.format(new_lr_vis))

            train_bar = tqdm(training_data_loader_cls)
            running_results = {'batch_sizes': 0,'cls_loss':0,'MVI_loss':0,'seg_loss':0}
            """   train    """
            # remove_all_file('debug')
            for low_resolution, high_resolution_Y_linerhigh, high_resolution,label in train_bar:

                count += 1
                batch_size = low_resolution.size(0)
                running_results['batch_sizes'] += batch_size
                seg.train()
                cls_net.train()
                MultiScale_Intergrate.train()
                high_resolution = Variable(high_resolution).cuda(device_ids[0])
                label = label.cuda(device_ids[0])

                ############################################
                #   training clsnet and vis parameters    #
                ############################################

                # obtain the segmentation map
                masks_pred = seg(high_resolution)  # [N,5,1024,1024]
                masks_pred=F.softmax(masks_pred)
                Seperate_lesion = masks_pred[:,0:4,:,:]  # [N,4,1024,1024]
                all_lesion= torch.unsqueeze((1-masks_pred[:,4,:,:]),1)  # [N,1,1024,1024]
                cls_input=torch.cat((high_resolution,Seperate_lesion),1)# [N,7,1024,1024]



                extractor = ModelOutputs(cls_net)
                features, fc_output = extractor(cls_input)
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

                vis_hr = MultiScale_Intergrate(cam_all_sr)  # [N,1,1024,1024]
                cls_output_sr = fc_output



                #  cls loss
                cls_loss = CE_criterion(cls_output_sr, label)
                MVI_loss = MSE_criterion(vis_hr, all_lesion)

                # update classification network
                optimizer_cls.zero_grad()
                cls_loss.backward(retain_graph=True)
                optimizer_cls.step()

                # update MultiScale_Intergrate network
                optimizer_MVI.zero_grad()
                # optimizer_seg.zero_grad()
                MVI_loss.backward()
                optimizer_MVI.step()
                # optimizer_seg.step()



                # loss for current batch before optimization
                running_results['cls_loss'] += cls_loss.item() * batch_size
                running_results['MVI_loss'] += MVI_loss.item() * batch_size
                running_results['seg_loss'] += MVI_loss.item() * batch_size


                train_bar.set_description(
                    desc='[%d/%d] cls_loss: %.4f MVI_loss: %.4f seg_loss: %.4f ' % (
                        epoch, Epoch,
                        running_results['cls_loss'] / running_results['batch_sizes'],
                        running_results['MVI_loss'] / running_results['batch_sizes'],
                        running_results['seg_loss'] / running_results['batch_sizes'],
                    ))

            """------------------Test--------------"""
            if epoch % 4 == 0 :
                print("Waiting Test!")
                with torch.no_grad():
                    count_val = 0
                    val_bar = tqdm(testing_data_loader_cls)
                    valing_results = {'batch_sizes': 0,'cls_acc':0,'kappa':0}
                    correct_all = 0
                    total_all = 0
                    correct_perclass = [0, 0, 0, 0, 0]
                    total_perclass = [0, 0, 0, 0, 0]
                    for low_resolution, high_resolution_linerhigh, high_resolution,label in val_bar:
                        batch_size = low_resolution.size(0)
                        valing_results['batch_sizes'] += batch_size
                        seg.eval()
                        cls_net.eval()
                        MultiScale_Intergrate.eval()
                        low_resolution = Variable(low_resolution).cuda(device_ids[0])
                        high_resolution = Variable(high_resolution).cuda(device_ids[0])
                        label = label.cuda(device_ids[0])
                        # SR metrics calculate
                        masks_pred = seg(high_resolution)  # [N,5,1024,1024]
                        masks_pred = F.softmax(masks_pred)
                        Seperate_lesion = masks_pred[:, 0:4, :, :]  # [N,4,1024,1024]
                        all_lesion = torch.unsqueeze((1 - masks_pred[:, 4, :, :]), 1)  # [N,1,1024,1024]
                        cls_input = torch.cat((high_resolution, Seperate_lesion), 1)  # [N,7,1024,1024]

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
                        val_bar.set_description(
                            desc='[Test!]OA: %.4f |kappa: %.4f '
                                 % (valing_results['cls_acc'], valing_results['kappa']))
                        count_val += 1

                torch.save(cls_net.state_dict(),
                           'model/model_' + model_name + '_' + dataset + '/cls_' + str(epoch + 1) + '.pth')
                torch.save(seg.state_dict(),
                           'model/model_' + model_name + '_' + dataset + '/seg_' + str(epoch + 1) + '.pth')
                torch.save(MultiScale_Intergrate.state_dict(),
                           'model/model_' + model_name + '_' + dataset + '/MultiScale_Intergrate' + str(epoch + 1) + '.pth')
                # save loss\scores\psnr\ssim
                if epoch != 0:
                    results['cls_loss'].append(running_results['cls_loss'] / running_results['batch_sizes'])
                    results['MVI_loss'].append(running_results['MVI_loss'] / running_results['batch_sizes'])
                    results['seg_loss'].append(running_results['seg_loss'] / running_results['batch_sizes'])
                else:
                    results['cls_loss'].append(0)
                    results['MVI_loss'].append(0)
                    results['seg_loss'].append(0)
                results['cls_acc'].append(valing_results['cls_acc'])
                results['kappa'].append(valing_results['kappa'])

                out_path = 'statistics/' + model_name + '_' + dataset + '/'
                data_frame = pd.DataFrame(
                    data={
                        'cls_loss': results['cls_loss'],
                          'MVI_loss': results['MVI_loss'],
                        'seg_loss': results['seg_loss'],
                          'cls_acc': results['cls_acc'], 'kappa': results['kappa'],
                          },
                )
                data_frame.to_csv(out_path + 'srf_' + str(upscale_factor) + '_train_results.csv',
                                  index_label='index')





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

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x_classifier = self.model._modules['module']._modules['denses'](x)

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
