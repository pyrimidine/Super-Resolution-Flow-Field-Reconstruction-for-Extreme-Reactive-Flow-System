import argparse
#%%
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor
from visual import *
from loss import *
from data import *
from visual import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=20, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=3, help='low to high resolution scaling factor')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--dic', type=str, default='./PIGAN_train1000_step200_depth1', help='folder to output model checkpoints')
parser.add_argument('--bi', type=bool, default=True, help='Binary image or not')
parser.add_argument('--bi_th', type=float, default=0.2, help='Binary threshold')
parser.add_argument('--depth', type=int, default=32, help='Binary threshold')

opt = parser.parse_args(args=[])
print(opt)

try:
    save_path = './' + opt.dic + '/test'
    os.makedirs(save_path)
    

except OSError:
    pass


path = r'.\dataset'
dataloader = test_load_data(path, 'test', opt)

generator = Generator(32, opt.upSampling)
generator = (torch.load(opt.dic + '/generator_final.pth'))

# print (generator)

content_criterion1 = nn.L1Loss()
content_criterion2 = SSIM()
physical_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()



#%%
print('Generator test')
f = open(os.path.join(save_path, 'test.txt'), "w")
mean_generator_content_loss = 0.0
mean_generator_physical_loss = 0.0

generator.eval()
for i, (p_LR, p_HR, pmax, rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t) in enumerate(dataloader):
    # Generate data

    # Generate real and fake inputs
    if opt.cuda:
        pmax = pmax.cuda()
        rho_HR = rho_HR.cuda()
        u_HR = u_HR.cuda()
        v_HR = v_HR.cuda()
        rho_HR_t = rho_HR_t.cuda()
        u_HR_t = u_HR_t.cuda()
        v_HR_t = v_HR_t.cuda()
        high_res_real = Variable(p_HR.cuda().to(torch.float32))
        high_res_fake1 = generator(Variable(p_LR).cuda().to(torch.float32))
        physical_residual = momentum_eqx(rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t, high_res_fake1, pmax)
        physical_residual = physical_residual/torch.max(physical_residual)

        err = (p_HR.detach().cpu().numpy() - high_res_fake1.detach().cpu().numpy())**2
        err = err/(p_HR.detach().cpu().numpy())**2


    ######### Train generator #########
    generator.zero_grad()
    # generator_content_loss = (1 - content_criterion2(high_res_fake, high_res_real))
    generator_content_loss = content_criterion1(high_res_fake1, high_res_real)

    generator.zero_grad()
    generator_physical_loss = physical_criterion(physical_residual, physical_residual*0)

    ######### Status and display #########
    sys.stdout.write('\r[%d/%d] Generator_Loss: %.4f Physics_Loss: %.4f Physics_Resudual: %.4f' % (i, len(dataloader), 
                                                                            generator_content_loss.item(), generator_physical_loss.item(), 
                                                                            np.mean(physical_residual.detach().cpu().numpy())))
    show_tensor_test(i, pmax, [p_LR, p_LR, high_res_real, high_res_fake1, (high_res_real-high_res_fake1)], path=save_path)
    print('\n', generator_content_loss.item(), generator_physical_loss.item(), np.mean(physical_residual.detach().cpu().numpy()), np.mean(err), file=f)
f.close


