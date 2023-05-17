#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
from visual import *
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from loss import *
import random
import numpy as np
from PIL import Image
import os
import random
import matplotlib
import matplotlib.pyplot as plt

class RDC_Dataset(Dataset):
    def __init__(self, path, flag='train', 
                 bi=False, bi_threshold=0.07,
                 depth=32,
                 LR_size=(20, 20), HR_size=(160, 160)):
        
        self.flag = flag
        self.bi = bi
        self.depth = depth
        self.path = os.path.join(path, flag)
        self.data_path_list = os.listdir(self.path)
        self.threshold = bi_threshold
        self.LR_size = LR_size
        self.HR_size = HR_size


    def __getitem__(self, index):
        # 根据索引返回数据
        [rho, u, v, p] = self.loader(self.data_path_list[index])
        pmax = torch.max(p)
        p = p/pmax
        rho_HR = self.HR_transform(rho)
        u_HR = self.HR_transform(u)
        v_HR = self.HR_transform(v)
        p_HR = self.HR_transform(p)
        p_LR = self.LR_transform(p)

        if index>0:
            [rho_t, u_t, v_t, p_t] = self.loader(self.data_path_list[index-1])
            rho_HR_t = self.HR_transform(rho_t)
            u_HR_t = self.HR_transform(u_t)
            v_HR_t = self.HR_transform(v_t)
        else:
            rho_HR_t = rho_HR
            u_HR_t = u_HR
            v_HR_t = v_HR
        
        p_LR = torch.unsqueeze(p_LR, dim=0)
        p_HR = torch.unsqueeze(p_HR, dim=0)
        rho_HR = torch.unsqueeze(rho_HR, dim=0)
        u_HR = torch.unsqueeze(u_HR, dim=0)
        v_HR = torch.unsqueeze(v_HR, dim=0)
        rho_HR_t = torch.unsqueeze(rho_HR_t, dim=0)
        u_HR_t = torch.unsqueeze(u_HR_t, dim=0)
        v_HR_t = torch.unsqueeze(v_HR_t, dim=0)

        return p_LR, p_HR, pmax, rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t

    def __len__(self):
        # 返回数据的长度
        return len(self.data_path_list)

    def loader(self, path):
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        data = torch.load(os.path.join(self.path, path))
        return data

    def LR_transform(self, data):
        data = torch.unsqueeze(data, dim=0)
        data = torch.unsqueeze(data, dim=0)
        data = F.interpolate(data, size=self.LR_size, mode='nearest')
        if self.bi == True:
            zero = torch.zeros_like(data)
            one = torch.ones_like(data)
            data = torch.where(data < self.threshold, zero, data)
            data = torch.where(data >= self.threshold, one, data)
        if self.depth != 32:
            data = (data*(2**self.depth-1)).int().float()
            data = data/(2**self.depth-1)
        data = torch.squeeze(data, dim=0)
        data = torch.squeeze(data, dim=0)
        return data

    def HR_transform(self, data):
        data = torch.unsqueeze(data, dim=0)
        data = torch.unsqueeze(data, dim=0)
        data = F.interpolate(data, size=self.HR_size, mode='nearest')
        data = torch.squeeze(data, dim=0)
        data = torch.squeeze(data, dim=0)
        return data


def load_data(path, flag, opt):

    LR = (opt.imageSize, opt.imageSize)
    HR = (opt.imageSize*(2**opt.upSampling), opt.imageSize*(2**opt.upSampling))
    train_data = RDC_Dataset(path, flag=flag, LR_size=LR, HR_size=HR, bi=opt.bi, bi_threshold=opt.bi_th, depth=opt.depth)

    train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, drop_last=True)

    return train_loader

def test_load_data(path, flag, opt):

    LR = (opt.imageSize, opt.imageSize)
    HR = (opt.imageSize*(2**opt.upSampling), opt.imageSize*(2**opt.upSampling))
    test_data = RDC_Dataset(path, flag=flag, LR_size=LR, HR_size=HR, bi=opt.bi, bi_threshold=opt.bi_th, depth=opt.depth)

    test_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, drop_last=True)

    return test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=20, help='the low resolution image size')
    parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
    parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
    parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--bi', type=bool, default=False, help='Binary image or not')
    parser.add_argument('--bi_th', type=float, default=0.1, help='Binary threshold')

    opt = parser.parse_args(args=[])
    print(opt)
    path = r'E:\RDE_GAN_HR_dataset\p=7e5_17e5\dataset'
    train_loader = load_data(path, flag='train', opt=opt)

    physical_criterion = nn.MSELoss()
    for epoch in range(5):
        for step, (p_LR, p_HR, pmax, rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t) in enumerate(train_loader):
            
            # show([p_LR, p_HR])

            print('Epoch: %d  Step: %d' %(epoch, step), p_LR.shape, p_HR.shape, rho_HR.shape, u_HR.shape, v_HR.shape)
            r1 = mass_eq(rho_HR, rho_HR_t, u_HR, v_HR)
            # r1 = momentum_eq(rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t, p_HR, pmax)
            r = r1.numpy()[0,0,:,:]
            loss = physical_criterion(r1, torch.zeros_like(r1))
            # show(r1/torch.max(r1))
            print('momentum_eq', loss.item(), torch.mean(torch.abs(r1)))
            if step == 0:
                break