import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from os import listdir
import torch
from torchvision.models.vgg import vgg16
from os import listdir
import torch.nn.functional as F
import torchvision.models as models
import torch.cuda
class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)




class KeNet(nn.Module):
    def __init__(self, classes_num=5):
        super(KeNet, self).__init__()
        resNet = models.resnet18(pretrained=True)
        resNet = list(resNet.children())[:-2]
        self.features = nn.Sequential(*resNet)

        self.attention = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.up_c2 = nn.Conv2d(1, 512, kernel_size=1, padding=0, bias=False)
        nn.init.constant_(self.up_c2.weight, 1)
        self.denses = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.features(x)

        atten_layers = self.attention(x)
        atten_layers = self.up_c2(atten_layers)
        # print atten_layers.shape
        mask_features = torch.matmul(atten_layers, x)
        # print mask_features.shape
        gap_features = F.avg_pool2d(mask_features, kernel_size=mask_features.size()[2:])
        # print gap_features.shape
        gap_mask = F.avg_pool2d(atten_layers, kernel_size=atten_layers.size()[2:])
        # print gap_mask.shape
        gap = torch.squeeze(Lambda(lambda x: x[0] / x[1])([gap_features, gap_mask]))
        # print gap.shape
        x = self.denses(gap)
        return x



class PerceptionLoss(nn.Module):
    def __init__(self,device_ids):
        super(PerceptionLoss, self).__init__()
        self.net = KeNet(classes_num=5).cuda(device_ids[0])
        self.pth =listdir('perceptual_model')
        self.net.load_state_dict(torch.load('perceptual_model/'+self.pth[0], map_location={'cuda:1': 'cuda:'+str(device_ids[0])}))
        self.perceptual_feature=[]

        # alex layer
        for Idx,Module_value in self.net._modules.items():
            if Idx != 'denses' and Idx != 'up_c2':
                if Idx=='attention':
                    for i in range(8):
                        self.perceptual_feature.append(self.net._modules['attention']._modules[str(i)])
                else:
                    self.perceptual_feature.append(Module_value)


        loss_network = nn.Sequential(*(self.perceptual_feature)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()
        self.upsampler = nn.Upsample(size=1024)

    def forward(self, high_resolution, fake_high_resolution):
        high_resolution = self.upsampler(high_resolution)
        fake_high_resolution = self.upsampler(fake_high_resolution)

        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss



class Seg_perceptual_loss(nn.Module):
    def __init__(self):
        super(Seg_perceptual_loss, self).__init__()




class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

