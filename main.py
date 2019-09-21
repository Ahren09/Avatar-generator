# coding:utf8
import os
import ipdb
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils import data
import matplotlib.pyplot as plt
import tqdm
from model import G, D
import time

from utils import Visualizer
from torchnet.meter import AverageValueMeter

IMG_SIZE = 96
BATCH_SIZE = 256
DATA_PATH = 'data/' # Relative path can be applied here
MODEL_SAVE_PATH = 'models/'
DIM_NOISE = 100
DIM_D = 64
DIM_G = 64
LR_G = 2e-4
LR_D = 2e-4
EPOCHS = 1000

MODEL_NUM_G = 4
MODEL_NUM_D = 4
PLOT_EVERY = 32
SAVE_EVERY = 20

DISCRIMINATE_EVERY = 4
GENERATE_EVERY = 4
CALCULATE_LOSS_EVERY = 8
PRETRAINED = True
env = 'GAN'

GEN_NUM = 2

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(IMG_SIZE),
        tv.transforms.CenterCrop(IMG_SIZE),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    vis = Visualizer(env)

    # drop_last: retain data that cannot suffice a batch
    dataset = tv.datasets.ImageFolder(DATA_PATH, transforms)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    netG, netD = G(DIM_NOISE, DIM_G), D(DIM_D)

    netG.to(device)
    netD.to(device)

    # Optimizers and Loss functions
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netG.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss().to(device)
    

    noises = torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    # Real imgs have 0 loss; fake imgs have 1
    true_labels = torch.ones(BATCH_SIZE).to(device)
    fake_labels = torch.zeros(BATCH_SIZE).to(device)

    max_epoch_G, max_epoch_D = 0, 0
    # Load pretrained network with greatest number of training epochs
    pretrained_root_g = MODEL_SAVE_PATH+'G/'
    if PRETRAINED and os.path.exists(pretrained_root_g):
        file_list = [file for file in os.listdir(pretrained_root_g) if file.endswith('.pth')]
        if file_list != []:
            index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
            max_epoch_G = max(index_list)
            model_name = 'model_g_%s.pth'%max_epoch_G
            print('Using mode:', model_name)
            netG.load_state_dict(torch.load(pretrained_root_g+model_name))
        else:
            print('Generator train from Step 0')

    pretrained_root_d = MODEL_SAVE_PATH+'D/'
    if PRETRAINED and os.path.exists(pretrained_root_d):
        file_list = [file for file in os.listdir(pretrained_root_d) if file.endswith('.pth')]
        if file_list != []:
            index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
            max_epoch_D = max(index_list)
            model_name = 'model_d_%s.pth'%max_epoch_D 
            print('Using mode:', model_name)
            netD.load_state_dict(torch.load(pretrained_root_d+model_name))
        else:
            print('Discriminator train from Epoch 0')

    for epoch in range(EPOCHS):
        time_start = time.time()
        for step, (img, _) in enumerate(dataloader):
            # Skip former steps if using pretrained models
            
            real_img = img.to(device)
            if step <= max_epoch_G and step <= max_epoch_D:
                continue

            # Train generator
            if step % GENERATE_EVERY == 0 and step > max_epoch_G:
                print('G - Epoch:',epoch,'| Step:',step)
                optimizer_G.zero_grad()
                noises.data.copy_(torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1)) # TODO: Why not noises=(torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1))
                fake_img = netG(noises)
                output_g = netD(fake_img)
                loss_g = criterion(output_g, true_labels)
                loss_g.backward()
                optimizer_G.step()
                errorg_meter.add(loss_g.item())
            
            if step % DISCRIMINATE_EVERY == 0 and step > max_epoch_D:
                # Identify real images
                print('D - Epoch:',epoch,'| Step:',step)
                output_d = netD(real_img)
                loss_real = criterion(output_d, true_labels)
                loss_real.backward()

                # Identify fake images
                optimizer_D.zero_grad()
                noises.data.copy_(torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1))
                fake_img = netG(noises)
                output_d = netD(fake_img)
                loss_fake = criterion(output_d, fake_labels)
                loss_fake.backward()
                optimizer_D.step()

                loss_d = loss_real + loss_fake
                errord_meter.add(loss_d.item())
                #fake_img = fake_img.detach()
            
            if (step+1) % PLOT_EVERY:
                fix_fake_imgs = netG(noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])



            
            
        time_end = time.time()
        print('Total time for epoch ',epoch,' is ',(time_end-time_start))
        if epoch and epoch % SAVE_EVERY == 0:
            torch.save(netG.state_dict(), MODEL_SAVE_PATH+'G/model_g_%s.pth'%step)
            torch.save(netD.state_dict(), MODEL_SAVE_PATH+'D/model_d_%s.pth'%step)
            errord_meter.reset()
            errorg_meter.reset()
         

def generate():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    noises = torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1).normal_(0, 1).to(device)
    netG, netD = G(DIM_NOISE, DIM_G).eval(), D(DIM_D).eval()

    netG.load_state_dict(torch.load(MODEL_SAVE_PATH+'G/model_g_%s.pth'%MODEL_NUM_G))
    netD.load_state_dict(torch.load(MODEL_SAVE_PATH+'D/model_d_%s.pth'%MODEL_NUM_D))

    netG.to(device)
    netD.to(device)

    fake_img = netG(noises)
    scores = netD(fake_img).detach()

    indexes = scores.topk(GEN_NUM)[1]
    results = []
    for i in indexes:
        results.append(fake_img.data[i])
    tv.utils.save_image(torch.stack(results), "result.png",  normalize=True, range=(-1, 1))

'''
def load_model(net, model_path, mode):
    pretrained_root = ''
    if mode == 'G':
        pretrained_root = model_path+'models/G/'
        if PRETRAINED and os.path.exists(pretrained_root):
            file_list = [file for file in os.listdir(pretrained_root) if file.endswith('.pth')]
            if file_list != []:
                index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
                max_epoch = max(index_list)
                model_name = 'model_g_%s.pth'%max_epoch
                print('Using mode:', model_name)
                net.load_state_dict(torch.load(pretrained_root+model_name))
            else:
                print('Generator train from Epoch 0')
    elif mode == 'D':
        pretrained_root = model_path+'models/D/':
        if PRETRAINED and os.path.exists(pretrained_root):
            file_list = [file for file in os.listdir(pretrained_root) if file.endswith('.pth')]
            if file_list != []:
                index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
                max_epoch = max(index_list)
                model_name = 'model_d_%s.pth'%max_epoch
                print('Using mode:', model_name)
                netG.load_state_dict(torch.load(pretrained_root+model_name))
            else:
                print('Discriminator train from Epoch 0')
    else:
        print("Error: Not valid mode")
'''


if __name__ == "__main__":

    train()
#generate()
