import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log2
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
from PIL import Image
import torch
import torch.optim as optim
import argparse
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import glob
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.models as models
import torch.cuda
from util import spatial_pyramid_pool
import os
from util import spatial_pyramid_pool
"""  --------------seg net ------------------"""




class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()

        self.MSconv1=nn.Conv2d(ch_in, int(ch_out//2), kernel_size=3, stride=1, padding=1, bias=True)
        self.MSconv2 = nn.Conv2d(ch_in, int(ch_out // 2), kernel_size=5, stride=1, padding=2, bias=True)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1=self.MSconv1(x)
        x2 = self.MSconv2(x)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x




class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up1=nn.Upsample(scale_factor=2)
        self.up2_1 = nn.Conv2d(ch_in, int(ch_out // 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.up2_2 = nn.Conv2d(ch_in, int(ch_out // 2), kernel_size=5, stride=1, padding=2, bias=True)
        self.up3 =nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x1 = self.up2_1(x)
        x2 = self.up2_2(x)
        x = torch.cat((x1, x2), 1)
        x = self.up3(x)
        return x

class Multiscale_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Multiscale_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out//3), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(ch_out//3)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out // 3), kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(int(ch_out // 3)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out // 3), kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(int(ch_out // 3)),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x1= self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x=torch.cat((x1,x2),1)
        x = torch.cat((x3, x), 1)
        return x



class Seg_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1,ratio=4):
        super(Seg_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16*ratio)
        self.Conv2 = conv_block(ch_in=16*ratio, ch_out=32*ratio)
        self.Conv3 = conv_block(ch_in=32*ratio, ch_out=64*ratio)
        self.Conv4 = conv_block(ch_in=64*ratio, ch_out=128*ratio)
        self.Conv5 = conv_block(ch_in=128*ratio, ch_out=256*ratio)

        self.Up5 = up_conv(ch_in=256*ratio, ch_out=128*ratio)
        self.Up_conv5 = conv_block(ch_in=256*ratio, ch_out=128*ratio)

        self.Up4 = up_conv(ch_in=128*ratio, ch_out=64*ratio)
        self.Up_conv4 = conv_block(ch_in=128*ratio, ch_out=64*ratio)

        self.Up3 = up_conv(ch_in=64*ratio, ch_out=32*ratio)
        self.Up_conv3 = conv_block(ch_in=64*ratio, ch_out=32*ratio)

        self.Up2 = up_conv(ch_in=32*ratio, ch_out=16*ratio)
        self.Up_conv2 = conv_block(ch_in=32*ratio, ch_out=16*ratio)

        self.Conv_1x1 = nn.Conv2d(16*ratio, output_ch, kernel_size=1, stride=1, padding=0)

        self.poly1 = nn.Conv2d(256*ratio, output_ch, kernel_size=1, stride=1, padding=0)
        self.poly2 = nn.Conv2d(128*ratio, output_ch, kernel_size=1, stride=1, padding=0)
        self.poly3 = nn.Conv2d(64*ratio, output_ch, kernel_size=1, stride=1, padding=0)
        self.poly4 = nn.Conv2d(32*ratio, output_ch, kernel_size=1, stride=1, padding=0)
        self.poly5 = nn.Conv2d(16*ratio, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x,eta1=0.4,eta2=0.2,eta3=0.2,eta4=0.2):


        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)

        self.out1_ori = self.poly1(x5)
        self.out2_ori = self.poly2(d5)
        self.out3_ori = self.poly3(d4)
        self.out4_ori = self.poly4(d3)
        self.out5_ori = self.poly5(d2)

        self.out1 = F.interpolate(self.out1_ori,size=(d2.shape[2], d2.shape[3]))
        self.out2 = F.interpolate(self.out2_ori,size=(d2.shape[2], d2.shape[3]))
        self.out3 = F.interpolate(self.out3_ori, size=(d2.shape[2], d2.shape[3]))
        self.out4 = F.interpolate(self.out4_ori,size=(d2.shape[2], d2.shape[3]))
        self.out5 = F.interpolate(self.out5_ori, size=(d2.shape[2], d2.shape[3]))

        self.out = eta1 * self.out1 + eta2 * self.out2 + eta3 * self.out3 + self.out4




        return self.out





"""  --------------SR net ------------------"""
from math import log2
import math




class res_(nn.Module):
    def __init__(self, in_channels=32, kernel=(3, 3), out_channels=32, stride=1):
        super(res_, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        y = self.relu1(self.conv1(x))
        return self.conv2(y) + x


class deconv_(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3, out_channels=32, stride=2):
        super(deconv_, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                padding=1, output_padding=1)
        self.relu1 = nn.LeakyReLU()

    def forward(self, x):
        y = self.relu1(self.deconv1(x))
        return y



class ISR_Net(torch.nn.Module):
    def __init__(self, channels=3, upsample_factor=4, device_ids=None,is_residual=False):
        super(ISR_Net, self).__init__()
        self.upsample_factor = upsample_factor
        self.device_ids = device_ids
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(), )
        for i in range(5):
            self.add_module('res_' + str(i + 1), res_())
        for i in range(int(log2(self.upsample_factor))):
            self.add_module('deconv_' + str(i + 1), deconv_())
        self.layer2 = nn.Conv2d(32, channels, kernel_size=(3, 3), stride=1, padding=1)
        self.is_residual=is_residual

    def forward(self, x, if_loop=False, lesion_seg_map=None, theta=0.5):  # x:  [N,3,32,32]
        x0 = self.layer1(x)
        x1 = self.__getattr__('res_' + str(1))(x0)
        x2 = self.__getattr__('res_' + str(2))(x1)
        x3 = self.__getattr__('res_' + str(3))(x2)
        x4 = self.__getattr__('res_' + str(4))(x3)
        x5 = self.__getattr__('res_' + str(5))(x4)

        if int(log2(self.upsample_factor)) == 3:
            x6 = self.__getattr__('deconv_' + str(1))(x5)
            x7 = self.__getattr__('deconv_' + str(2))(x6)
            x8 = self.__getattr__('deconv_' + str(3))(x7)
            x = self.layer2(x8)
            if not self.is_residual:
                # x = torch.nn.Sigmoid()(x)
                x = (torch.tanh(x) + 1) / 2
            return x

        if int(log2(self.upsample_factor)) == 2:
            x6 = self.__getattr__('deconv_' + str(1))(x5)
            x7 = self.__getattr__('deconv_' + str(2))(x6)
            x = self.layer2(x7)
            if not self.is_residual:
                # x = torch.nn.Sigmoid()(x)
                x = (torch.tanh(x) + 1) / 2
            return x
        if int(log2(self.upsample_factor)) == 1:
            x6 = self.__getattr__('deconv_' + str(1))(x5)

            x = self.layer2(x6)
            if not self.is_residual:
                # x = torch.nn.Sigmoid()(x)
                x = (torch.tanh(x) + 1) / 2
            return x


    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



"""  --------------CLS net ------------------"""

import cv2
from torch.autograd import Variable
from torchvision import models

class KeNet_v1(nn.Module):
    def __init__(self, classes_num=3):
        super(KeNet_v1, self).__init__()
        self.resNet1 = models.resnet18(pretrained=True)
        self.resNet = list(self.resNet1.children())[:-2]
        self.features = nn.Sequential(*self.resNet)
        self.new_first_layer=nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.denses = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, classes_num)
        )
        self.dense_spp = nn.Sequential(
            nn.Linear(10752, 512),
            nn.Dropout(0.5),
            nn.Linear(512, classes_num)
        )

    def forward(self, x):
        x = self.new_first_layer(x)
        x = self.features._modules['1'](x)
        x = self.features._modules['2'](x)
        x = self.features._modules['3'](x)
        x = self.features._modules['4'](x)
        x = self.features._modules['5'](x)
        x = self.features._modules['6'](x)
        x = self.features._modules['7'](x)
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(x.size(0), -1)
        # x_classifier = self.denses(x)
        spp = spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)
        x_classifier = self.dense_spp(spp)
        return  x_classifier


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.gradients = [[],[],[],[]]
        self.target_activations = [[], [], [], []]
        self.dense_spp = nn.Sequential(
            nn.Linear(10752, 512),
            nn.Dropout(0.5),
            nn.Linear(512, classes_num)
        )

    def save_gradient_256(self, grad):
        self.gradients[0].append(grad)

    def save_gradient_128(self, grad):
        self.gradients[1].append(grad)

    def save_gradient_64(self, grad):
        self.gradients[2].append(grad)

    def save_gradient_32(self, grad):
        self.gradients[3].append(grad)


    def __call__(self, x):  # x~[B,3,128,128]

        x =self.model._modules['module']._modules['new_first_layer'](x)
        # x = self.model._modules['module']._modules['features']._modules['0'](x)
        x = self.model._modules['module']._modules['features']._modules['1'](x)
        x = self.model._modules['module']._modules['features']._modules['2'](x)
        x = self.model._modules['module']._modules['features']._modules['3'](x)
        x = self.model._modules['module']._modules['features']._modules['4'](x)
        x.register_hook(self.save_gradient_256)
        self.target_activations[0] += [x]
        x = self.model._modules['module']._modules['features']._modules['5'](x)
        x.register_hook(self.save_gradient_128)
        self.target_activations[1] += [x]
        x = self.model._modules['module']._modules['features']._modules['6'](x)
        x.register_hook(self.save_gradient_64)
        self.target_activations[2] += [x]
        x = self.model._modules['module']._modules['features']._modules['7'](x)
        x.register_hook(self.save_gradient_32)
        self.target_activations[3] += [x]
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(x.size(0), -1)
        spp = spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)
        fc_output = self.dense_spp(spp)
        # fc_output = self.model._modules['module']._modules['denses'](x)



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
        self.model.zero_grad()
        one_hot_all.backward(retain_graph=True)



        for i in range(4):

            grads_val = self.gradients[i][-1]
            target = self.target_activations[i][-1]
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




        return cam_all_hr, fc_output


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







class Multi_Intergrate(nn.Module):
    def __init__(self):
        super(Multi_Intergrate, self).__init__()
        self.conv1 = nn.Sequential(#nn.BatchNorm2d(2),
                                   nn.Conv2d(2, 1, kernel_size=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(#nn.BatchNorm2d(2),
                                   nn.Conv2d(2, 1, kernel_size=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(#nn.BatchNorm2d(2),
                                   nn.Conv2d(2, 1, kernel_size=1),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, cam_all):
        cam_256 = cam_all['size256']
        cam_128 =cam_all['size128']
        cam_64 = cam_all['size64']
        cam_32 = cam_all['size32'] # [N,32,32]

        for i in range(len(cam_32)):
            cam_32_temp = Compose([ToPILImage(), Resize(cam_32[i].shape[0] * 2, interpolation=Image.BICUBIC), ToTensor()])(cam_32[i].cpu())
            cam_64_temp= torch.unsqueeze(cam_64[i].cpu(), 0)
            if i == 0:
                cam_32_upx2 = cam_32_temp
                cam_64_upx1= cam_64_temp
            else:
                cam_32_upx2 = torch.cat((cam_32_upx2, cam_32_temp), 0)
                cam_64_upx1 = torch.cat((cam_64_upx1, cam_64_temp), 0)
        cam_64_cat_32 =torch.cat((torch.unsqueeze(cam_32_upx2,1),torch.unsqueeze(cam_64_upx1,1)),1)   # [N,2,64,64]
        cam_64_new = self.conv1(cam_64_cat_32.cuda(cam_32[0].device.index))  # [N,1,64,64]
        cam_64_new = cam_64_new.cpu()


        for i in range(cam_64_new.shape[0]):
            cam_64_temp = torch.unsqueeze(
                Compose([ToPILImage(), Resize(cam_64_new.shape[2] * 2, interpolation=Image.BICUBIC), ToTensor()])(
                    cam_64_new[i][0]), 0)
            cam_128_temp=torch.unsqueeze(cam_128[i].cpu(), 0)
            if i == 0:
                cam_64_upx2 = cam_64_temp
                cam_128_upx1 = cam_128_temp
            else:
                cam_64_upx2 = torch.cat((cam_64_upx2, cam_64_temp), 0)
                cam_128_upx1 = torch.cat((cam_128_upx1, cam_128_temp), 0)
        cam_128_cat_64 = torch.cat((cam_64_upx2,torch.unsqueeze(cam_128_upx1,1)),1)  # [N,2,128,128]
        cam_128_new = self.conv2(cam_128_cat_64.cuda(cam_64[0].device.index))  # [N,1,128,128]
        cam_128_new = cam_128_new.cpu()


        for i in range(cam_128_new.shape[0]):
            cam_128_temp = torch.unsqueeze(
                Compose([ToPILImage(), Resize(cam_128_new.shape[2] * 2, interpolation=Image.BICUBIC), ToTensor()])(
                    cam_128_new[i][0]), 0)
            cam_256_temp = torch.unsqueeze(cam_256[i].cpu(), 0)
            if i == 0:
                cam_128_upx2 = cam_128_temp
                cam_256_upx1 = cam_256_temp
            else:
                cam_128_upx2 = torch.cat((cam_128_upx2, cam_128_temp), 0)
                cam_256_upx1 = torch.cat((cam_256_upx1, cam_256_temp), 0)
        cam_256_cat_128 = torch.cat((cam_128_upx2,torch.unsqueeze(cam_256_upx1,1)),1)  # [N,2,128,128]
        cam_256_new = self.conv3(cam_256_cat_128.cuda(cam_128[0].device.index))  # [N,1,256,256]

        for i in range(cam_256_new.shape[0]):
            if i == 0:
                cam_256_temp = Compose([ToPILImage(), Resize(1024, interpolation=Image.BICUBIC), ToTensor()])(cam_256_new[i][0].cpu())
            else:
                cam_256_temp = torch.cat((Compose([ToPILImage(), Resize(1024, interpolation=Image.BICUBIC), ToTensor()])(cam_256_new[i][0].cpu()), cam_256_temp), 0)
        cam_256_new = torch.unsqueeze(cam_256_temp, 1).cuda(cam_256[0].device.index)






        # for i in range(len(cam_256)):
        #     cam_256_temp = Compose([ToPILImage(), Resize(1024, interpolation=Image.BICUBIC), ToTensor()])(cam_256[i].cpu())
        #
        # cam_256_new = torch.unsqueeze(cam_256_temp, 1).cuda(cam_256[0].device.index)



        return cam_256_new












