import argparse, os
import pdb
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
from model_wgan_bicubic64 import _netGL, _netDL,_netGH, _netDH, L1_Charbonnier_loss
from dataset import FastLoader
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
import time

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DGGAN WGAN")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument('--lrG', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--lrD', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument("--step", type=int, default=60, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=32, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

def main(dataname,dataid):

    global opt, model 
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = FastLoader('../../datasets/NYU/pkls/nyu_%d.pkl'%dataid)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print('===> Building generator model')
    netG_HR = _netGH()
    netG_LR = _netGL()
    print('===> Building discriminator model')    
    netD_HR = _netDH()
    netD_LR = _netDL()

    print('===> Loading VGG model') 
    model_urls = {
        "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    }

    netVGG = models.vgg19()
    netVGG.load_state_dict(model_zoo.load_url(model_urls['vgg19']))

    weight = torch.FloatTensor(64,1,3,3)
    parameters = list(netVGG.parameters())
    for i in range(64):
        weight[i,:,:,:] = parameters[0].data[i].mean(0)
    bias = parameters[1].data

    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.conv = conv2d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.feature = nn.Sequential(*list(netVGG.features.children())[1:-1])
            self._initialize_weights()

        def forward(self, x):
            out = self.conv(x)
            out = self.feature(out)
            return out

        def _initialize_weights(self):
            self.conv.weight.data.copy_(weight)
            self.conv.bias.data.copy_(bias)

    netContent = _content_model()

    print('===> Building Loss')
    criterion = L1_Charbonnier_loss()

    print("===> Setting GPU")
    if cuda:
        netG_HR = netG_HR.cuda()
        netD_HR = netD_HR.cuda()
        netG_LR = netG_LR.cuda()
        netD_LR = netD_LR.cuda()
        netContent = netContent.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG_HR.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG_HR.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizerDL = optim.RMSprop(netD_LR.parameters(), lr = opt.lrD)
    optimizerDH = optim.RMSprop(netD_HR.parameters(), lr = opt.lrD)
    optimizerGL = optim.RMSprop(netG_LR.parameters(), lr = opt.lrG)
    optimizerGH = optim.RMSprop(netG_HR.parameters(), lr = opt.lrG)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        e_start_time = time.time()
        train(training_data_loader, optimizerGH, optimizerDH,optimizerGL, optimizerDL, netG_HR, netD_HR,netG_LR, netD_LR, netContent, criterion, epoch)
        save_checkpoint(netG_HR, epoch, dataid)
        e_end_time = time.time()
        elapse_time = e_end_time-e_start_time
        print('Time = %.5fs/Epoch'%elapse_time)

def adjust_learning_rate(optimizer, epoch, type):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if type == 'G':
        lr = opt.lrG * (0.1 ** (epoch // opt.step))
    elif type == 'D':
        lr = opt.lrD * (0.1 ** (epoch // opt.step))
    return lr

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    BATCH_SIZE = real_data.shape[0]
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)
    ag2 = real_data.nelement()//BATCH_SIZE
    # ag2 must be int
    alpha = alpha.expand(BATCH_SIZE, ag2)
    alpha = alpha.contiguous()
    # alpha = alpha.view(BATCH_SIZE, 3, 32, 32)
    # single channel
    # print(real_data.shape)
    # print(fake_data.shape)
    h,w = real_data.shape[2],real_data.shape[3]
    alpha = alpha.view(BATCH_SIZE, 1, h, w)
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train(training_data_loader, optimizerGH, optimizerDH,optimizerGL, optimizerDL, netG_HR, netD_HR,netG_LR, netD_LR, netContent, criterion, epoch):
    lrGH = adjust_learning_rate(optimizerGH, epoch-1,'G')
    lrGL = adjust_learning_rate(optimizerGL, epoch-1,'G')
    lrDH = adjust_learning_rate(optimizerDH, epoch-1,'D')
    lrDL = adjust_learning_rate(optimizerDL, epoch-1,'D')
    for param_group in optimizerGH.param_groups:
        param_group["lr"] = lrGH
    for param_group in optimizerGL.param_groups:
        param_group["lr"] = lrGL
    for param_group in optimizerDH.param_groups:
        param_group["lr"] = lrDH
    for param_group in optimizerDL.param_groups:
        param_group["lr"] = lrDL
    netG_HR.train()
    netD_HR.train()
    netG_LR.train()
    netD_LR.train()

    one = torch.FloatTensor([1.])
    mone = one * -1
    content_weight = torch.FloatTensor([0.1])
    # content_weight = torch.FloatTensor([1.])
    adversarial_weight = torch.FloatTensor([1.])
    # adversarial_weight = torch.FloatTensor([0.01])
    for iteration, batch in enumerate(training_data_loader, 1):

        LR, HR, Ld, Hd = batch[0], batch[1], batch[2], batch[3]
        # print(HR.shape,LR.shape)
        d_ratio = (Hd/Ld).mean()
        d_scale = torch.sigmoid(d_ratio)
        # d_scale = F.tanh(d_ratio/4)
        # d_scale = F.softmax(d_ratio)
        # d_scale = d_ratio/4
        # print(d_scale)
        # 用dscale约束每次迭代的参数更新幅度
        # for param_group in optimizerGH.param_groups:
        #     param_group["lr"] = lrGH*d_scale
        # for param_group in optimizerGL.param_groups:
        #     param_group["lr"] = lrGL*d_scale
        # for param_group in optimizerDH.param_groups:
        #     param_group["lr"] = lrDH*d_scale
        # for param_group in optimizerDL.param_groups:
        #     param_group["lr"] = lrDL*d_scale   
        if opt.cuda:
            LR = LR.cuda()
            HR = HR.cuda()
            one, mone, content_weight, adversarial_weight = one.cuda(), mone.cuda(), content_weight.cuda(), adversarial_weight.cuda()

        ############################
        # (1) Update D network: loss = D(x)) - D(G(z))
        ###########################
        # netD_HR --->fake HR and real HR
        # netD_LR --->fake LR and real LR
        # netG_HR ---> fake LR--> fake HR
        # netG_LR ---> HR---> fake LR

        errD_real_l = netD_LR(LR)
        errD_real_l.backward(one, retain_graph=True)
        input_G_hr = HR.data
        fake_LR = netG_LR(input_G_hr).data
        errD_fake_l = netD_LR(fake_LR)
        errD_fake_l.backward(mone)

        # train with gradient penalty
        gradient_penalty_l = calc_gradient_penalty(netD_LR, LR.data, fake_LR)
        gradient_penalty_l.backward()
        errD_l = errD_real_l - errD_fake_l + gradient_penalty_l
        Wasserstein_D_l  = errD_real_l - errD_fake_l
        optimizerDL.step()
        # for p in netD_LR.parameters(): # reset requires_grad
        #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        netD_LR.zero_grad()
        netG_LR.zero_grad()
        netContent.zero_grad()

        # train with real
        errD_real_h = netD_HR(HR)
        errD_real_h.backward(one, retain_graph=True)
        # train with fake
        input_G_lr = fake_LR.data
        fake_HR = netG_HR(input_G_lr).data
        errD_fake_h = netD_HR(fake_HR)
        errD_fake_h.backward(mone)
        gradient_penalty_h = calc_gradient_penalty(netD_HR, HR.data, fake_HR)
        gradient_penalty_h.backward()
        errD_h = errD_real_h - errD_fake_h + gradient_penalty_h
        Wasserstein_D_h = errD_real_h - errD_fake_h
        optimizerDH.step()

        # for p in netD_HR.parameters(): # reset requires_grad
        #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        netD_HR.zero_grad()
        netG_HR.zero_grad()
        netContent.zero_grad()
        
        ############################
        # (2) Update G network: loss = D(G(z))
        ###########################      
        #----------Part1 Downsample-------------
        fake_D_lr = netG_LR(HR)
        # content_fake_lr = netContent(fake_D_lr)
        # content_real_lr = netContent(LR)
        # content_real_lr = Variable(content_real_lr.data)       
        # content loss for this task???
        # content_loss_lr = criterion(content_fake_lr, content_real_lr)
        # content_loss_lr.backward(content_weight, retain_graph=True)
        # content_loss_l = content_loss_lr

        adversarial_loss_l = netD_LR(fake_D_lr)
        adversarial_loss_l.backward(adversarial_weight)

        optimizerGL.step()
        netD_LR.zero_grad()
        netG_LR.zero_grad()
        # netContent.zero_grad()
        #---- cycle 1 distribution up_real_lr-->real HR
        fake_D_lr_up = netG_HR(LR)
        adversarial_loss_l_up = netD_HR(fake_D_lr_up)
        adversarial_loss_l_up.backward(adversarial_weight)

        optimizerGH.step()
        netD_HR.zero_grad()
        netG_HR.zero_grad()
        #--------Part2 Upsample-----------
        fake_D_lr = netG_LR(HR)
        fake_D_x2 = netG_HR(fake_D_lr)
        content_fake_x2 = netContent(fake_D_x2)
        content_real_x2 = netContent(HR)
        content_real_x2 = Variable(content_real_x2.data)       
        content_loss_x2 = criterion(content_fake_x2, content_real_x2)
        content_loss_x2.backward(content_weight, retain_graph=True)
        content_loss = content_loss_x2

        adversarial_loss = netD_HR(fake_D_x2)
        lossG = adversarial_loss+content_loss
        adversarial_loss.backward(adversarial_weight)

        optimizerGH.step()
        netD_HR.zero_grad()
        netG_HR.zero_grad()
        netContent.zero_grad()
        #---- cycle 2 fake HR->fake LR
        fake_D_lr = netG_LR(HR)
        fake_D_x2 = netG_HR(fake_D_lr)        
        fake_D_hr_down = netG_LR(fake_D_x2)
        content_fake_hr_down = netContent(fake_D_hr_down)
        content_real_lr = netContent(fake_D_lr)
        content_real_lr = Variable(content_real_lr.data)       
        content_loss_down_lr = criterion(content_fake_hr_down, content_real_lr)
        content_loss_down_lr.backward(content_weight, retain_graph=True)
        content_loss_dl = content_loss_down_lr

        adversarial_loss_dl = netD_LR(fake_D_hr_down)
        adversarial_loss_dl.backward(adversarial_weight)

        optimizerGL.step()
        netD_LR.zero_grad()
        netG_LR.zero_grad()
        netContent.zero_grad()
        # print network and loss
        if iteration%5 == 0:
            print("===> Epoch[{}]({}/{}): LossD: {:.5f} [{:.5f} - {:.5f}] LossG: {:.5f} [{:.5f} + {:.5f}]".format(
                epoch, iteration, len(training_data_loader), 
                errD_h.item(), errD_real_h.item(), errD_fake_h.item(), lossG.item(), adversarial_loss.item(), content_loss.item()))   
    
def save_checkpoint(model, epoch,dataid):
    model_folder = './86_zssr/model_%d/'%dataid
    model_out_path = model_folder + "biwgan_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    utils.save_experiment()
    # ims = [98,80,3,7,5,1,15,2,4,40,28,49,8,26,27,65,11,81,91,55,67,22,25,85,57,52,48,61,83,62,56,45,
    # 50,44,18,35,43,97,86,0,88,58,30,87,78,74,99,29,75,64]
    ims = [50,44,18,35,43,97,86,0,88,58,30,87,78,74,99,29,75,64]
    # for i in range(86,87):
    #     main('NYU',i)
    for i in range(len(ims)):
        main('NYU',ims[i])
