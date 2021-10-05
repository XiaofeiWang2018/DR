import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from sklearn.metrics import auc
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

# import torchsnooper
# @torchsnooper.snoop()
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, device_ids,window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).cuda(device_ids[0])
    return _ssim(img1, img2, window, window_size, channel, size_average)


range_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
def computeConfMatElements(thresholded_proba_map, ground_truth):
    P = np.count_nonzero(ground_truth)
    TP = np.count_nonzero(thresholded_proba_map * ground_truth)
    FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map * ground_truth))

    return P, TP, FP
def computeAUPR(proba_map, ground_truth, threshold_list):
    proba_map = proba_map.astype(np.float32)
    proba_map = proba_map.reshape(-1)
    ground_truth = ground_truth.reshape(-1)
    precision_list_treshold = []
    recall_list_treshold = []
    # loop over thresholds
    for threshold in threshold_list:
        # threshold the proba map
        thresholded_proba_map = np.zeros(np.shape(proba_map))
        thresholded_proba_map[proba_map >= threshold] = 1
        # print(np.shape(thresholded_proba_map)) #(400,640)

        # compute P, TP, and FP for this threshold and this proba map
        P, TP, FP = computeConfMatElements(thresholded_proba_map, ground_truth)

        # check that ground truth contains at least one positive
        if (P > 0 and (TP + FP) > 0):
            precision = TP * 1. / (TP + FP)
            recall = TP * 1. / P
        else:
            precision = 1
            recall = 0

        # average sensitivity and FP over the proba map, for a given threshold
        precision_list_treshold.append(precision)
        recall_list_treshold.append(recall)

    # aupr = 0.0
    # for i in range(1, len(precision_list_treshold)):
    #     aupr = aupr + precision_list_treshold[i] * (recall_list_treshold[i] - recall_list_treshold[i - 1])
    precision_list_treshold.append(1)
    recall_list_treshold.append(0)
    return auc(recall_list_treshold, precision_list_treshold)

