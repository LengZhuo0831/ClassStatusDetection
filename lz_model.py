
import torch
from models import ResNet,GlobalAveragePooling

import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
import cv2
from torchvision.transforms import ToTensor,Normalize,Compose,AutoAugmentPolicy,AutoAugment
from torchvision.models import resnet18
import re
import random
from torch.optim import SGD
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F


class LZ_Mouth_Dataset(Dataset):
    def __init__(self,mode='eye',train=True):
        super(LZ_Mouth_Dataset, self).__init__()
        assert mode in ['eye','mouth']
        self.mode = mode
        self.files = []
        self.label_dict = dict()
        for file in os.listdir('data/'+mode+'_close'):
            filepath = 'data/' + mode + '_close/' + file
            self.files.append(filepath)
            self.label_dict[filepath] = 1 if self.mode=='eye' else 0

        for file in os.listdir('data/'+mode+'_open'):
            filepath = 'data/' + mode + '_open/' + file
            self.files.append(filepath)
            self.label_dict[filepath] = 0 if self.mode=='eye' else 1

        self.transform = Compose(
            [
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                ToTensor(),
                Normalize(mean=[123.675/255, 116.28/255, 103.53/255], std=[58.395/255, 57.12/255, 57.375/255])
            ]
        ) if train else Compose(
            [
                ToTensor(),
                Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],std=[58.395 / 255, 57.12 / 255, 57.375 / 255])
            ]
        )

        # 随机选取 80 %
        random.seed(0)
        random.shuffle(self.files)
        if train:
            self.data_infos = self.files[:-len(self.files)//5]
        else:
            self.data_infos = self.files[-len(self.files) // 5:]


    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        img = Image.open(self.data_infos[idx])
        img = img.convert('RGB')
        img = self.transform(img)
        label = self.label_dict[self.data_infos[idx]]
        label = torch.tensor(label)

        return img, label


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EmotionResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(EmotionResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class LZ_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LZ_CNN, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNet(18,3,32,32)
        self.global_pooling = GlobalAveragePooling()
        self.classifier = nn.Sequential(nn.Linear(256,num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pooling(x)[0]
        x = self.classifier(x)
        x = x.softmax(-1)
        return x

class LZ_Trainer:
    def __init__(self,mode='mouth',model_name = 'LZ_CNN'):
        if mode=='mouth':
            self.trn_dataset = LZ_Mouth_Dataset('../data/mouth_clip/',train=True)
            self.test_dataset = LZ_Mouth_Dataset('../data/mouth_clip/',train=False)
        elif mode=='eye':
            self.trn_dataset = LZ_Mouth_Dataset('../data/eye_clip/', train=True)
            self.test_dataset = LZ_Mouth_Dataset('../data/eye_clip/', train=False)
        self.trn_dataloader = DataLoader(self.trn_dataset,128,True)
        self.test_dataloader = DataLoader(self.test_dataset,1,False)

        self.model = eval(model_name)(2)
        self.loss  = nn.CrossEntropyLoss()
        self.cuda = torch.cuda.is_available()
        self.mode = mode
        if self.cuda:
            self.model.cuda()


    def train(self):
        start_epoch = 0
        end_epoch = 40
        initial_lr = 0.01
        optimizer = SGD(self.model.parameters(),lr=initial_lr)
        cur_acc = 0
        for epoch in range(start_epoch,end_epoch):
            print(f"training epoch: {epoch}/{end_epoch}")
            self.model.train()
            for i,(img,label) in enumerate(self.trn_dataloader):
                if self.cuda:
                    img = img.cuda()
                    label = label.cuda()
                out = self.model(img)
                loss = self.loss(out,label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                out = out.argmax(1)
                res = out * 2 + label
                res = res.cpu().numpy()
                res = np.unique(res, return_counts=True)
                res_dict = {0: 0, 1: 0, 2: 0, 3: 0}  # tn,fn,fp,tp
                for k, idx in enumerate(res[0]):
                    res_dict[idx] += res[1][k]
                tn, fn, fp, tp = res_dict[0],res_dict[1],res_dict[2],res_dict[3]
                print(f"iter {i}, loss:{loss.item()}, acc:{(tp+tn)/(tp+tn+fp+fn)}, precision:{tp/(tp+fp) if (tp+fp)!=0 else 'NAN'}, recall:{tp/(tp+fn) if (tp+fn)!=0 else 'NAN'}")

            # val
            res_dict={0:0,1:0,2:0,3:0}
            self.model.eval()
            for i,(img,label) in enumerate(self.test_dataloader):
                if self.cuda:
                    img = img.cuda()
                    label = label.cuda()
                out = self.model(img)
                out = out.argmax(1)
                res = out*2+label
                res = res.cpu().numpy()
                res = np.unique(res,return_counts=True)
                for k,idx in enumerate(res[0]):
                    res_dict[idx]+=res[1][k]
            tn, fn, fp, tp = res_dict[0], res_dict[1], res_dict[2], res_dict[3]
            print(f"val epoch {epoch}, acc:{(tp + tn) / (tp + tn + fp + fn)}, precision:{tp/(tp+fp) if (tp+fp)!=0 else 'NAN'}, recall:{tp/(tp+fn) if (tp+fn)!=0 else 'NAN'}")

            # save
            acc = (tp + tn) / (tp + tn + fp + fn)
            if acc>cur_acc:
                torch.save(self.model.state_dict(),f'checkpoint/{self.mode}_best.pth')

    def test(self, state_dict_path=None):
        self.model.eval()
        if state_dict_path!=None:
            self.model.load_state_dict(torch.load(state_dict_path))
        elif self.mode=='eye':
            self.model.load_state_dict(torch.load('checkpoint/eye_best.pth'))
        else:
            self.model.load_state_dict(torch.load('checkpoint/mouth_best.pth'))

        res_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        for i, (img, label) in enumerate(self.test_dataloader):
            if self.cuda:
                img = img.cuda()
                label = label.cuda()
            out = self.model(img)
            out = out.argmax(1)
            res = out * 2 + label
            res = res.cpu().numpy()
            res = np.unique(res, return_counts=True)
            for i, idx in enumerate(res[0]):
                res_dict[idx] += res[1][i]
        tn, fn, fp, tp = res_dict[0], res_dict[1], res_dict[2], res_dict[3]
        print(f"test results\nacc:{(tp + tn) / (tp + tn + fp + fn)}, precision:{tp/(tp+fp) if (tp+fp)!=0 else 'NAN'}, recall:{tp/(tp+fn) if (tp+fn)!=0 else 'NAN'}")


