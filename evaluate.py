#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/5/20
@Description:
"""
import os
import torch
import argparse
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torch.utils import data
from networks.resnet import resnet18
from dataloader import DatasetSamplerCoeff
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--img_dir', default="../datasets/RAFdb_224/test", help='testset dir')
parser.add_argument('--model_path', default="saved_models/RAFDB_pretrain_resnet_1124/testbest.pth", help='saved model path')
parser.add_argument('--rat', action='store_true', help='use rat block?')
parser.add_argument('--coeff', action='store_true', help='use coeff?')
parser.add_argument('--cuda', type=str, default="0", help='use cuda')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([T.ToTensor()])
test_dataset = DatasetSamplerCoeff(opt.img_dir, transform=transform, mode='test')
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1, shuffle=False)

model = resnet18(pretrained=False, num_classes=7, cfg=opt)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load(opt.model_path)
model_state_dict = checkpoint['state_dict']
model.load_state_dict(model_state_dict)

cudnn.benchmark = True

# classes = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
classes = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral','Contempt']
def plot_confusion_matrix(cm,
                          save_name=None,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    plt.figure(figsize=(12, 8))    #
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    plt.tight_layout()      # 自动调整子图参数，使之填充整个图像区域

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
        print("image saved in", save_name)
    plt.show()


def get_emotion_score(matrix):
    macc = []
    for i in range(len(matrix)):
        macc.append(matrix[i][i])
    return np.array(macc)


model.eval()

crt = 0
num = 0
prediction, truth = [],[]
print("waiting...")
for i, (img, target, coeff) in enumerate(test_loader):
    num += 1

    input_var = torch.autograd.Variable(img)
    pred_score = model(input_var, coeff)

    predict_label = torch.argmax(pred_score).item()
    label = target.item()

    prediction.append(predict_label)
    truth.append(label)

    if predict_label == label:
        crt += 1

matrix = confusion_matrix(truth, prediction)
print(matrix)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

macc = get_emotion_score(matrix)
print(macc)
print("Mean Acc:", macc.mean())
acc = crt/num
print("Sum Acc:", crt, num, acc)

plot_confusion_matrix(matrix, save_name="matrix", title="ResNet18 RAF-DB (Acc:{:.2f}%)".format(acc*100))
