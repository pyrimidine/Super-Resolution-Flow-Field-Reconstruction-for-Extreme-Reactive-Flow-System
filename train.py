#!/usr/bin/env python
#%%
import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
from models import *
from data import *
from visual import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=20, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=3, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--lastTrain', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--physicalWeights', type=float, default=1e-5, help="PINN balance weight")
parser.add_argument('--out', type=str, default='./checkPoints', help='folder to output model checkpoints')
parser.add_argument('--bi', type=bool, default=True, help='Binary image or not')
parser.add_argument('--bi_th', type=float, default=0.2, help='Binary threshold')
parser.add_argument('--depth', type=int, default=32, help='')


opt = parser.parse_args(args=[])
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

path = r'.\dataset\'
dataloader = load_data(path, 'train', opt)

generator = Generator(64, opt.upSampling)
if opt.lastTrain != '':
    generator = torch.load(os.path.join(opt.lastTrain, 'generator_final.pth'))
discriminator = Discriminator()
if opt.lastTrain != '':
    discriminator = torch.load(os.path.join(opt.lastTrain, 'discriminator_final.pth'))
finetune_net = Finetune_net()

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
# print (feature_extractor)
content_criterion1 = nn.L1Loss()
content_criterion2 = SSIM()
physical_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    finetune_net.cuda()
    feature_extractor.cuda()
    content_criterion1.cuda()
    content_criterion2.cuda()
    physical_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)
optim_finetune = optim.Adam(finetune_net.parameters(), lr=opt.generatorLR)




print('SR pre-training')
for epoch in range(1):
    mean_generator_content_loss = 0.0
    mean_generator_total_loss = 0.0


    for i, (p_LR, p_HR, pmax, rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t) in enumerate(dataloader):

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
            # physical_residual = momentum_eq(rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t, high_res_fake1, pmax)
            # physical_residual = physical_residual/1e+9

            # target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            # target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
        

        ######### Train generator #########
        generator.zero_grad()

        generator_content_loss = content_criterion1(high_res_fake1, high_res_real)
        # generator_physical_loss = physical_criterion(physical_residual, physical_residual*0)
        
        # finetune_content_loss = content_criterion1(SA, physical_residual)
        # mean_finetune_total_loss += finetune_content_loss.item()
        # generator_total_loss += finetune_content_loss

        generator_total_loss = generator_content_loss 

        mean_generator_total_loss += generator_total_loss.item()

        generator_total_loss.backward()
        optim_generator.step()
        # optim_finetune.step()
        
        ######### Status and display #########
        sys.stdout.write('\r[ %d %d ][ %d %d ] Generator_Loss: %.4f' % (epoch, opt.nEpochs, i, len(dataloader), generator_content_loss.item()))

    if (epoch+1)%1 == 0:
        show_tensor(epoch, [p_LR, high_res_real, high_res_fake1], path=os.path.join('./', opt.out))

    sys.stdout.write('\r[%d %d][%d %d] Generator_Loss: %.4f\n' % (epoch, opt.nEpochs, i, len(dataloader), mean_generator_content_loss/len(dataloader)))

    # Do checkpointing
    torch.save(generator, '%s/generator_final.pth' % opt.out)




#%%
# SRGAN training
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR*0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1)
f = open(os.path.join('%s/training_out.txt' % opt.out), "w")
print('generator_content_loss, generator_physical_loss, physical_residual', file=f)

print('SRGAN training')
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    mean_finetune_total_loss = 0.0
    mean_generator_physical_loss = 0.0

    for i, (p_LR, p_HR, pmax, rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t) in enumerate(dataloader):

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
            # high_res_fake, SA = finetune_net(high_res_fake1)
            physical_residual = momentum_eq(rho_HR, u_HR, v_HR, rho_HR_t, u_HR_t, v_HR_t, high_res_fake1, pmax)
            # physical_residual = physical_residual/torch.max(physical_residual)
            physical_residual = physical_residual/1e+9

            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
        
        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake1.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.item()
        
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()
        finetune_net.zero_grad()

        real_features = Variable(feature_extractor(high_res_real.repeat(1,3,1,1)).data)
        fake_features = feature_extractor(high_res_fake1.repeat(1,3,1,1))

        generator_content_loss = content_criterion1(high_res_fake1, high_res_real) + 0.006*content_criterion1(fake_features, real_features)
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake1), ones_const)
        generator_physical_loss = physical_criterion(physical_residual, physical_residual*0)
        
        # finetune_content_loss = content_criterion1(SA, physical_residual)
        # mean_finetune_total_loss += finetune_content_loss.item()
        # generator_total_loss += finetune_content_loss

        generator_total_loss = generator_content_loss + 1e-2*generator_adversarial_loss 
        generator_total_loss += 1e-1*generator_physical_loss

        mean_generator_content_loss += generator_content_loss.item()
        mean_generator_adversarial_loss += generator_adversarial_loss.item()
        mean_generator_physical_loss += generator_physical_loss.item()
        mean_generator_total_loss += generator_total_loss.item()

        generator_total_loss.backward()
        optim_generator.step()
        # optim_finetune.step()
        
        ######### Status and display #########
        sys.stdout.write('\r[ %d %d ][ %d %d ] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f %.4f %.4f Physics_Loss: %.4f' % (epoch, opt.nEpochs, i, len(dataloader),
        discriminator_loss.item(), generator_content_loss.item(), generator_adversarial_loss.item(), generator_total_loss.item(), generator_physical_loss.item()))

    if (epoch+1)%1 == 0:
        show_tensor(epoch, [p_LR, high_res_real, high_res_fake1, physical_residual], path=os.path.join('./', opt.out))

    sys.stdout.write('\r[%d %d][%d %d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f %.4f %.4f Physics_Loss: %.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
    mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader), mean_generator_physical_loss/len(dataloader)))

    # Do checkpointing
    torch.save(generator, '%s/generator_final.pth' % opt.out)
    torch.save(finetune_net, '%s/finetune_final.pth' % opt.out)
    torch.save(discriminator, '%s/discriminator_final.pth' % opt.out)
    print('\n', generator_content_loss.item(), generator_physical_loss.item(), np.mean(physical_residual.detach().cpu().numpy()), file=f)
f.close

#%%
# Avoid closing
while True:
    pass

# %%
