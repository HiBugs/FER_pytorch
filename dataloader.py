#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/11/4
@Description:
"""
import os
import cv2
import numpy as np

from PIL import Image
import torch.utils.data as data

Emotion_Dir = ['0Surprise', '1Fear', '2Disgust', '3Happy', '4Sad', '5Angry', '6Neutral']
# Emotion_Dir = ['0Surprise', '1Fear', '2Disgust', '3Happy', '4Sad', '5Angry', '6Neutral', '7Contempt']
Emotion_GAN_Dir = ['0Surprise', '1Fear', '2Disgust', '4Sad', '5Angry']


def get_files(path, suffix='.h5', mode='img'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if mode == 'img':
                if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.png':
                    file_list.append(os.path.join(root, f))
            else:
                if os.path.splitext(f)[1] == suffix:
                    file_list.append(os.path.join(root, f))

    return file_list

def load_data(data_dir):
    image_list, label_list, coeff_list = [], [], []
    for idx, emd in enumerate(Emotion_Dir):
        path = os.path.join(data_dir, emd)
        img_names = get_files(path, mode='img')
        for img in img_names:
            image_list.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
            with open(img.replace('.jpg', '_Coeff.txt'), "r") as f:
                x = f.readline()
                exp_coeff = list(map(np.float32, x.split(" ")[:-1]))
                coeff_list.append(exp_coeff)
            label_list.append(idx)
    return np.asarray(image_list), np.asarray(label_list), np.asarray(coeff_list)

def brightness_3d(image, min=0.5, max=2.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)

    hsv_img = hsv[:, :, 2]
    mask = hsv_img * random_br > 255
    v_channel = np.where(mask, 255, hsv_img * random_br)

    hsv[:, :, 2] = v_channel
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

class DatasetSamplerCoeff(data.Dataset):
    def __init__(self, img_dir, transform=None, mode='test'):
        self.images, self.labels, self.coeff = load_data(img_dir)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        img = self.images[index]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        if self.mode == 'train':
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)

            if np.random.random() > 0.5:
                img = brightness_3d(img)

            angle = 0
            scale = 1
            if np.random.random() > 0.5:
                angle = np.random.random_integers(-10, 10)
            if np.random.random() > 0.5:
                scale = np.random.random_integers(80, 120)/100
            matRotate = cv2.getRotationMatrix2D((112, 112), angle, scale)  # mat rotate 1 center 2 angle 3 缩放系数
            img = cv2.warpAffine(img, matRotate, (224, 224))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        img = Image.fromarray(img, mode='RGB')
        img = self.transform(img)

        return img, self.labels[index], self.coeff[index]

    def __len__(self):
        return len(self.labels)
