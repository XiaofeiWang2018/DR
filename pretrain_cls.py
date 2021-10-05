
import torch.nn as nn
import torch.optim as optim
import argparse
from data_process import DRDataset
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import torch.nn.functional as F
import torch
import argparse
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import torchvision.models as models
import torch.cuda
from tqdm import tqdm
from network import cls_net
from loss import LabelSmoothingCrossEntropy
from config.cls_v1 import *
from sklearn.metrics import precision_score, cohen_kappa_score,recall_score,roc_auc_score,f1_score




# hyper parameters
EPOCH = 500
num_epochs_celoss=25
num_epochs_decay=160
img_size=256
num_class=5


if 1:
    device_ids = [2]
    basic_net='resnet' # 'resnet' 'densenet'  'vgg' 'squeezenet'
    if basic_net == 'resnet':
        net = KeNet_v1(classes_num=num_class).cuda(device_ids[0])
    elif basic_net == 'densenet' or basic_net == 'densenet_spp':
        net = KeNet_v2(classes_num=num_class).cuda(device_ids[0])
    elif basic_net == 'vgg':
        net = KeNet_v3(classes_num=num_class).cuda(device_ids[0])
    elif basic_net == 'alexnet':
        net = KeNet_v4(classes_num=num_class).cuda(device_ids[0])
    elif basic_net == 'inception':
        net = KeNet_v5(classes_num=num_class).cuda(device_ids[0])
    elif basic_net == 'squeezenet':
        net = KeNet_v6(classes_num=num_class).cuda(device_ids[0])
    net = torch.nn.DataParallel(net, device_ids)  # multi-GPUs
    model_name='Oriset_'+basic_net+'_scale'+str(img_size)
    dataset='DDR'
    num_thread= 8
    train_BATCH_SIZE = 16
    test_BATCH_SIZE =1

# net.load_state_dict(torch.load('model/model_Oriset_resnet_scale512_DDR/net_257.pth'))

def reload_net(path):
    trainednet = torch.load(path)
    return trainednet


def main():

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(device_ids[0]))
    criterion_MSE=nn.MSELoss().cuda(device_ids[0])
    optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=weight_decay)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=4e-08)
    # if optimizer_choice=='SGD':
    #     optimizer = optim.SGD(params=net.parameters(), lr=LR,momentum=momentum,weight_decay=weight_decay)
    # elif optimizer_choice=='Adam':
    #     optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
    # if lr_scheduler_choice=='Plateau':
    #     scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',min_lr=min_lr,factor=factor, patience=patience)
    # elif lr_scheduler_choice=='Cosine':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=min_lr)

    dr_dataset_train = DRDataset(root_img='../data/DDR/Grading/Original_Images/Train/img_before_ori',
                                 root_label='../data/DDR/Grading/Groundtruths/Train/',
                                 img_size=img_size, num_class=num_class, transform=True,
                                 # root_img1='../data/DDR/Grading/Original_Images/Test/img_before_ori',
                                 # root_label1='../data/DDR/Grading/Groundtruths/Test/'
                                 )
    dr_dataset_test = DRDataset(
        root_img='../data/DDR/Grading/Original_Images/Test/img_before_ori',
        root_label='../data/DDR/Grading/Groundtruths/Test/', img_size=img_size,
        num_class=num_class, transform=False)

    loader_train = DataLoader(dr_dataset_train, batch_size=train_BATCH_SIZE, num_workers=num_thread, shuffle=True)
    loader_test_tbX = DataLoader(dr_dataset_test, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=True)
    loader_test = DataLoader(dr_dataset_test, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=False)
    writer = SummaryWriter(logdir='runs/runs_'+model_name+'_'+dataset)
    count_all=0
    new_lr = LR
    with open('acc/acc_'+model_name+'_'+dataset+'.txt', "w+") as f:
        for epoch in range(EPOCH):
            if (epoch + 1) > (EPOCH - num_epochs_decay):
                new_lr -= (LR / float(num_epochs_decay))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print('Decay learning rate to lr: {}.'.format(new_lr))
            running_results = { 'acc': 0, 'acc_loss': 0}
            # print('Decay learning rate to lr: {}.'.format(optimizer.param_groups[0]['lr']))
            train_bar = tqdm(loader_train)
            count = 0
            """--------------------------------------Train---------------------------------------"""
            for packs in train_bar:
                count+=1
                count_all+=1
                net.train()
                inputs, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(device_ids[0])
                optimizer.zero_grad()

                outputs = net(inputs)
                loss_ce = criterion(outputs, labels) # vanilla softmax loss
                _, predicted = torch.max(outputs.data, 1)
                loss_mse=criterion_MSE(predicted.float(),labels.float())
                if epoch<num_epochs_celoss:
                    loss=loss_ce
                else:
                    loss = loss_ce+loss_mse
                loss.backward()
                optimizer.step()

                total = labels.size(0)
                correct = predicted.eq(labels.data).cpu().sum()

                running_results['acc'] += 100. * correct / total
                running_results['acc_loss'] += loss.item()

                train_bar.set_description(
                    desc=model_name + ' [%d/%d] acc_loss: %.4f  | Acc: %.4f' % (
                        epoch, EPOCH,
                        running_results['acc_loss'] / count,
                        running_results['acc'] / count
                    ))




                """------------------tensorboard test--------------"""
                if count % 4 == 0:
                    writer.add_scalar('scalar/train_loss_per_iter', loss.item(), count_all)
                    writer.add_scalar('scalar/acc_batchwise', (100. * correct / total), count_all)
                if count % 45 ==0:
                    with torch.no_grad():
                        (inputs, labels) = iter(loader_test_tbX).next()
                        net.eval()
                        inputs, labels = inputs.cuda(device_ids[0]), labels.cuda(device_ids[0])
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss_tbX_ce = criterion(outputs, labels)
                        loss_tbX_mse = criterion_MSE(predicted.float(), labels.float())
                        if epoch < num_epochs_celoss:
                            loss_tbX = loss_tbX_ce
                        else:
                            loss_tbX = loss_tbX_ce + loss_tbX_mse
                        writer.add_scalar('scalar/test_ce_loss', loss_tbX.item(), count_all)
            #

            """------------------Test--------------"""

            if epoch % 4 == 0:
                test_bar = tqdm(loader_test)
                print("Waiting Test!")
                with torch.no_grad():
                    correct_all = 0
                    total_all = 0
                    correct_perclass=[0, 0, 0, 0, 0]
                    total_perclass = [0, 0, 0, 0, 0]
                    GT_label=[]
                    prediction_label=[]
                    for packs in test_bar:
                        net.eval()
                        images, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(device_ids[0])
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total_all += labels.size(0)
                        correct_all += (predicted == labels).sum()
                        labels = labels.cpu().numpy()
                        predicted = predicted.cpu().numpy()
                        for i_test in range(test_BATCH_SIZE):
                            total_perclass[labels[i_test]]+=1
                            if predicted[i_test]==labels[i_test]:
                                correct_perclass[labels[i_test]]+=1
                            ###
                            GT_label.append(labels[i_test])
                            prediction_label.append(predicted[i_test])

                    GT_label=np.array(GT_label)
                    prediction_label = np.array(prediction_label)
                    correct_all= correct_all.cpu().numpy()
                    OA = 100. * correct_all / total_all
                    Acc_0 = 100. * correct_perclass[0]/total_perclass[0]
                    Acc_1 = 100. * correct_perclass[1] / total_perclass[1]
                    Acc_2 = 100. * correct_perclass[2] / total_perclass[2]
                    Acc_3 = 100. * correct_perclass[3] / total_perclass[3]
                    Acc_4 = 100. * correct_perclass[4] / total_perclass[4]




                    kappa_linear =100. *cohen_kappa_score(y1=GT_label,y2=prediction_label,weights='linear')
                    kappa_quadratic =100. *cohen_kappa_score(y1=GT_label,y2=prediction_label,weights='quadratic')


                    precision_weighted = 100. *precision_score(y_true=GT_label, y_pred=prediction_label, average='weighted')
                    fscor_weighted = 100. *f1_score(y_true=GT_label, y_pred=prediction_label, average='weighted')




                    print('Test EPOCH=%03d | OA=：%.3f%% ' % (epoch + 1,OA))

                    if OA>70:
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % ('model/model_'+model_name+'_'+dataset, epoch + 1))
                    f.write("EPOCH=%03d | OA=：%.3f%% | kappa_l=：%.3f%% | kappa_q=：%.3f%% | pre_w=：%.3f%% | fscore_w=：%.3f%% |"
                            " Acc_0=：%.3f%% | Acc_1=：%.3f%% | Acc_2=：%.3f%% | Acc_3=：%.3f%% | Acc_4=：%.3f%%"
                            % (epoch + 1, OA,kappa_linear,kappa_quadratic,precision_weighted,fscor_weighted,Acc_0,Acc_1,Acc_2,Acc_3,Acc_4))
                    f.write('\n')
                    f.flush()





        writer.close()


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)




if __name__ == "__main__":


    init_seed = 1115
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    if not os.path.isdir('model/model_'+model_name+'_'+dataset):
        os.makedirs('model/model_'+model_name+'_'+dataset)
    else:
        remove_all_file('model/model_'+model_name+'_'+dataset)
    if os.path.isdir('runs/runs_'+model_name+'_'+dataset):
        remove_all_file('runs/runs_' + model_name + '_' + dataset)

    main()

