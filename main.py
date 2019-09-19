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

IMG_SIZE = 96
BATCH_SIZE = 64
DATA_PATH = 'data/' # Relative path can be applied here
MODEL_SAVE_PATH = 'models/'
DIM_NOISE = 100
DIM_D = 64
DIM_G = 64
LR_G = 2e-4
LR_D = 2e-4
EPOCHS = 50

MODEL_NUM_G = 1
MODEL_NUM_D = 1
SAVE_EVERY = 100

DISCRIMINATE_EVERY = 4
GENERATE_EVERY = 4
CALCULATE_LOSS_EVERY = 8
PRETRAINED = True

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(IMG_SIZE),
        tv.transforms.CenterCrop(IMG_SIZE),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

    # Real imgs have 0 loss; fake imgs have 1
    true_labels = torch.ones(BATCH_SIZE).to(device)
    fake_labels = torch.zeros(BATCH_SIZE).to(device)

    max_epoch = 0
    # Load pretrained network with greatest number of training epochs
    pretrained_root_g = MODEL_SAVE_PATH+'G/'
    if PRETRAINED and os.path.exists(pretrained_root_g):
        file_list = [file for file in os.listdir(pretrained_root_g) if file.endswith('.pth')]
        if file_list != []:
            index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
            max_epoch = max(index_list)
            model_name = 'model_g_%s.pth'%max_epoch
            print('Using mode:', model_name)
            netG.load_state_dict(torch.load(pretrained_root_g+model_name))
        else:
            print('Generator train from Step 0')

    pretrained_root_d = MODEL_SAVE_PATH+'D/'
    if PRETRAINED and os.path.exists(pretrained_root_d):
        file_list = [file for file in os.listdir(pretrained_root_d) if file.endswith('.pth')]
        if file_list != []:
            index_list = [int(file.split('.pth')[0].split('_')[2]) for file in file_list]
            model_name = 'model_d_%s.pth'%max_epoch
            print('Using mode:', model_name)
            netD.load_state_dict(torch.load(pretrained_root_d+model_name))
        else:
            print('Discriminator train from Epoch 0')
    
    plt.ion()

    for epoch in range(EPOCHS):
        
        loss, loss_list = 0, []
        plt.cla()
        for step, (img, _) in enumerate(dataloader):
            # Skip former steps if using pretrained models
            if PRETRAINED and step < max_epoch:
                continue
            print('Epoch:',epoch,'| Step:',step)
            real_img = img.to(device)

            # Train generator
            if step % GENERATE_EVERY == 0:
                optimizer_G.zero_grad()
                noises.data.copy_(torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1)) # TODO: Why not noises=(torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1))
                fake_img = netG(noises)
                output_g = netD(fake_img)
                loss_g = criterion(output_g, true_labels)
                loss_g.backward()
                optimizer_G.step()
                loss += loss_g
            
            if step % DISCRIMINATE_EVERY == 0:
                # Identify real images
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

                loss += loss_real + loss_fake
                #fake_img = fake_img.detach()

            if (step+1) % CALCULATE_LOSS_EVERY == 0:
                avg_loss = loss.data.numpy()/CALCULATE_LOSS_EVERY
                print('Loss:', avg_loss)
                loss_list.append(avg_loss)
                loss = 0
                plt.ylim((1,3))
                plt.plot(range(int(step/CALCULATE_LOSS_EVERY)+1), loss_list, 'r-')
                plt.draw()
                plt.pause(0.01)
            if (step+1) % SAVE_EVERY == 0:
                torch.save(netG.state_dict(), MODEL_SAVE_PATH+'G/model_g_%s.pth'%step)
                torch.save(netD.state_dict(), MODEL_SAVE_PATH+'D/model_d_%s.pth'%step)

            
        
         

def generate():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    noises = torch.randn(BATCH_SIZE, DIM_NOISE, 1, 1).to(device)
    netG, netD = G(DIM_NOISE, DIM_G).eval(), D(DIM_D).normal_(0, 1)

    netG.load_state_dict(torch.load(MODEL_SAVE_PATH+'G/model_g_%s.pth'%MODEL_NUM_G))
    netD.load_state_dict(torch.load(MODEL_SAVE_PATH+'D/model_d_%s.pth'%MODEL_NUM_D))

    fake_img = netG(noises)
    scores = netD(fake_img).detach()

    indexes = scores.topk(opt.gen_num)[1]
    results = []
    for i in indexes:
        results.append(fake_img.data[i])
    tv.utils.save_image(torch.stack(results), normalize=True, range=(-1, 1))

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