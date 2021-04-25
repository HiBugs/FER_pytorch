#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/11/4
@Description:
"""
import argparse
import time
from progress.bar import Bar
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
import torch
import os
import numpy as np
import shutil
import cv2
import random
from dataloader import DatasetSamplerCoeff
from networks.resnet import resnet18
from utils import cal_accuracy, AverageMeter
parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--train_dir', default="../datasets/RAFdb_224/train", help='trainset dir')
parser.add_argument('--val_dir', default="../datasets/RAFdb_224/train", help='valset dir')
parser.add_argument('--test_dir', default="../datasets/RAFdb_224/test", help='testset dir')
parser.add_argument('--save_path', default="saved_models/RAFDB_coeff_1222", help='saved models dir')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=128, help='testing batch size')
parser.add_argument('--num_epochs', type=int, default=120, help='the starting epoch count')
parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
parser.add_argument('--not_pretrain', action='store_true', help='use fr pretrain?')
parser.add_argument('--rat', action='store_true', help='use rat block?')
parser.add_argument('--coeff', action='store_true', help='use coeff?')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=666')
parser.add_argument('--cuda', type=str, default="0", help='use cuda')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# np.random.seed(opt.seed)
# torch.manual_seed(opt.seed)
# torch.cuda.manual_seed_all(opt.seed)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    name = os.path.join(opt.save_path, filename)
    torch.save(state, name)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # lr = 0.0001 if epoch < 10 else 0.0001 * np.exp(0.1 * (10 - epoch))
    # if epoch>=10:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.0001 * np.exp(0.1 * (10 - epoch))
    if epoch in [int(opt.num_epochs * 0.3), int(opt.num_epochs * 0.6), int(opt.num_epochs * 0.9)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def validate(val_loader, model):
    top1 = AverageMeter()
    model.eval()

    for i, (img, target, coeff) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(img)
        pred_score = model(input_var, coeff)
        prec1 = cal_accuracy(pred_score.data, target, topk=(1,))
        top1.update(prec1[0], img.size(0))

    return top1.avg


def train(train_loader, model, criterion,  optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    num_iters = len(train_loader)
    bar = Bar('{}/{}'.format(epoch, opt.num_epochs), max=num_iters)

    for i, (img, target, coeff) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(img)
        target_var = torch.autograd.Variable(target)

        pred_score = model(input_var, coeff)

        loss = criterion(pred_score, target_var)

        prec1 = cal_accuracy(pred_score.data, target, topk=(1,))
        losses.update(loss.item(), img.size(0))

        top1.update(prec1[0], img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                    i+1, num_iters, phase='train',
                    total=bar.elapsed_td, eta=bar.eta_td)
        Bar.suffix = Bar.suffix + '| loss: {:.4f} | acc: {:.4f}'.format(losses.avg, top1.avg)
        bar.next()

    bar.finish()
    return losses.avg, top1.avg


transform = T.Compose([T.ToTensor()])
train_dataset = DatasetSamplerCoeff(opt.train_dir, transform=transform, mode='train')
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=opt.batch_size,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=True)

val_dataset = DatasetSamplerCoeff(opt.val_dir, transform=transform, mode='test')
val_loader = data.DataLoader(val_dataset,
                             batch_size=opt.test_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

# test_dataset = ImageFolder(test_image_dir, transform2)
test_dataset = DatasetSamplerCoeff(opt.test_dir, transform=transform, mode='test')
test_loader = data.DataLoader(test_dataset,
                              batch_size=opt.test_batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

model = resnet18(pretrained=False, num_classes=7, cfg=opt)
model = torch.nn.DataParallel(model).cuda()

if not opt.not_pretrain:
    checkpoint = torch.load('models/pretrain_models/ijba_res18_naive.pth.tar')
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
            pass
        else:
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict, strict=False)


cudnn.benchmark = True
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr,
                            momentum=0.9,
                            weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()

val_best_acc = 0
stop_num = 0
test_best_acc = 0

for epoch in range(opt.num_epochs):
    adjust_learning_rate(optimizer, epoch)

    train_loss, train_acc = train(train_loader, model, criterion,  optimizer, epoch)

    val_acc = validate(val_loader, model)
    test_acc = validate(test_loader, model)

    if test_acc >= test_best_acc:
        test_best_acc = test_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_prec': test_best_acc,
        }, "testbest.pth")

    if val_acc >= val_best_acc:
        val_best_acc = val_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_prec': val_best_acc,
        }, "trainbest.pth")

    print(" * Evaluate * ValAcc: {:.4f} | BestValAcc: {:.4f} || TestAcc: {:.4f} | BestTestAcc: {:.4f}".
          format(val_acc.item(), val_best_acc.item(), test_acc.item(), test_best_acc.item()))
