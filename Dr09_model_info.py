# ========================================================================
# 선일 제품라벨의 CNN 학습 코드 : Localization + Classification
# 작성일 : 2024-01-05
# 수정사항 : Datasets_cnnf_2 -> train & test로 양분 (val 삭제))
#          Attention + transformation
# ========================================================================
import copy
import os
import sys
import time
import datetime as dt
import warnings

import numpy as np      # linear algebra
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.optim import SGD, Adam, lr_scheduler
import torchvision

from torchsummary import summary
from torchinfo import summary as summary2
from thop import profile
from myutils import load_model


warnings.filterwarnings('ignore')

# 난수 시드 설정
random.seed(10)      # 10 -> 5
torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)

# 전역변수 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 데이터셋 준비 ################################################################
IMG_H, IMG_W = 480, 640     # STN
# IMG_H, IMG_W = 224, 224     # 기존 모델 학습
mean = (0.242, 0.175, 0.141)  # for Sunil datasets
std = (0.147, 0.132, 0.118)


def conv_block(num_in, num_out, kernel_size, padding=0, stride=1, dropout=0):
    """컨볼루션 레이어 = Conv2d + BatchNorm2d + ReLU + MaxPool2d"""
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(num_out),    # BatchNorm에 bias가 포함되어 있으므로 conv2d는 bias=False
        nn.ReLU(),
        nn.MaxPool2d(2))
        
        
 # Depthwise Separable Convolution
def dsconv_block(num_in, num_out, kernel_size, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(num_in, num_in, kernel_size, stride=stride, padding=padding, groups=num_in, bias=False),
        # nn.BatchNorm2d(num_in),  # groups=num_in : 각 입력 채널이 독립적으로 처리 (=DSConv)
        # nn.ReLU(),
        nn.Conv2d(num_in, num_out, kernel_size=1, bias=False),  # groups=1 (default) : 모든 입력 채널이 출력 채널과 연결)
        nn.BatchNorm2d(num_out),                                # groups=num_in : 각 입력 채널이 독립적으로 처리 (=DSConv)
        nn.ReLU(),                                              # 입력 채널 수와 출력 채널 수가 같아야 함
        # nn.Sigmoid(),
        nn.MaxPool2d(2))


class LocalizerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 7, padding=3),        # 1/2 = (320, 240)
            conv_block(16, 32, 5, padding=2),       # 1/4 = (160, 120)
            conv_block(32, 48, 3, padding=1),       # 1/8 = (80, 60)
            conv_block(48, 60, 3, padding=1),       # 1/16 = (40, 30)
            conv_block(60, 72, 3, padding=1),       # 1/32 = (20, 15)
            conv_block(72, 84, 3, padding=1),       # 1/64 -> (10, 7)
            conv_block(84, 96, 3, padding=1)        # 1/128 -> (5, 3)
        )

        # [3 * 2] 크기의 아핀(affine) 행렬에 대해 예측
        # self.dropout = nn.Dropout(p=_dropout)
        self.fc = nn.Sequential(
            nn.Linear(1440, 64),    # W * H * C = 5 * 3 * 96 = 1440
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Dropout(_dropout),
            nn.Linear(64,  2),
            # nn.Sigmoid()      # [0, 1] 범위로 만들기 위해서
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        # x = self.dropout(x)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


class ClassifierNet(nn.Module):
    def __init__(self, roi_size=64, c_dropout=0, l_dropout=0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = nn.Dropout(p=c_dropout)
        self.fc = nn.Linear(256, 4)

        self.localizer = LocalizerNet()
        self.ncxy = None
        # self.theta = None
        self.roi = None
        self.roi_size = roi_size
        
        # self.theta = None
        # 추론용: batch_size(1), dype 수정요망
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cuda:0')
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float16, device='cuda:0')
        self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cpu')
        self.theta[:, 0, 0] = self.roi_size / 640
        self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

    def forward(self, x):
        # theta = self.localizer(x)
        self.ncxy = self.localizer(x)
        # self.theta = theta.view(-1, 2, 3)
        # self.theta[:, 0, 2] = 2 * self.theta[:, 0, 2] - 1   # tx, ty => [-1, 1]
        # self.theta[:, 1, 2] = 2 * self.theta[:, 1, 2] - 1   # tx, ty => [-1, 1]

        # # 변환행렬 생성
        self.theta[:, :, 2] = self.ncxy
        # self.theta = torch.zeros(x.size(0), 2, 3, dtype=torch.float, device=x.device)
        # self.theta[:, 0, 2] = 2 * self.ncxy[:, 0] - 1  # tx, ty => [-1, 1]
        # self.theta[:, 1, 2] = 2 * self.ncxy[:, 1] - 1
        # self.theta[:, 0, 0] = self.roi_size / 640
        # self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

        grid = F.affine_grid(self.theta, torch.Size((x.size(0), 3, 96, 96)), align_corners=False)
        self.roi = F.grid_sample(x, grid, align_corners=False)

        x = self.features(self.roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
        


class LocalizerNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),        # 1/2 = (320, 240)
            conv_block(16, 32, 3, padding=1),       # 1/4 = (160, 120)
            conv_block(32, 48, 3, padding=1),       # 1/8 = (80, 60)
            conv_block(48, 60, 3, padding=1),       # 1/16 = (40, 30)
            conv_block(60, 72, 3, padding=1),       # 1/32 = (20, 15)
            conv_block(72, 84, 3, padding=1),       # 1/64 -> (10, 7)
            conv_block(84, 96, 3, padding=1)        # 1/128 -> (5, 3)
        )

        # [3 * 2] 크기의 아핀(affine) 행렬에 대해 예측
        # self.dropout = nn.Dropout(p=_dropout)
        self.fc = nn.Sequential(
            nn.Linear(1440, 64),    # W * H * C = 5 * 3 * 96 = 1440
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Dropout(_dropout),
            nn.Linear(64,  2),
            # nn.Sigmoid()      # [0, 1] 범위로 만들기 위해서
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        # x = self.dropout(x)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


class ClassifierNetV2(nn.Module):
    def __init__(self, roi_size=64, c_dropout=0, l_dropout=0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = nn.Dropout(p=c_dropout)
        self.fc = nn.Linear(256, 4)

        self.localizer = LocalizerNetV2()
        self.ncxy = None
        self.theta = None
        self.roi = None
        self.roi_size = roi_size
        
        # self.theta = None
        # 추론용: batch_size(1), dype 수정요망
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cuda:0')
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float16, device='cuda:0')
        self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cpu')
        self.theta[:, 0, 0] = self.roi_size / 640
        self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

    def forward(self, x):
        self.ncxy = self.localizer(x)
        # self.theta = theta.view(-1, 2, 3)
        # self.theta[:, 0, 2] = 2 * self.theta[:, 0, 2] - 1   # tx, ty => [-1, 1]
        # self.theta[:, 1, 2] = 2 * self.theta[:, 1, 2] - 1   # tx, ty => [-1, 1]

        # # 변환행렬 생성
        self.theta[:, :, 2] = self.ncxy
        # self.theta = torch.zeros(x.size(0), 2, 3, dtype=torch.float, device=x.device)
        # self.theta[:, 0, 2] = 2 * self.ncxy[:, 0] - 1  # tx, ty => [-1, 1]
        # self.theta[:, 1, 2] = 2 * self.ncxy[:, 1] - 1
        # self.theta[:, 0, 0] = self.roi_size / 640
        # self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

        grid = F.affine_grid(self.theta, torch.Size((x.size(0), 3, 96, 96)), align_corners=False)
        self.roi = F.grid_sample(x, grid, align_corners=False)

        x = self.features(self.roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
        
        
class OnlyClassifierNet(nn.Module):
    def __init__(self, roi_size=64, c_dropout=0, l_dropout=0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = nn.Dropout(p=c_dropout)
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        # x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


class ClassifierNetVf(nn.Module):
    def __init__(self, roi_size=64, in_size=96, c_dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = nn.Dropout(p=c_dropout)
        self.fc = nn.Linear(256, 4)

        # self.localizer = LocalizerNetV2(_dropout=l_dropout)
        self.localizer = LocalizerNet3Dsconv()
        self.ncxy = None
        self.roi = None
        self.roi_size = roi_size
        self.in_size = in_size
        
        # self.theta = None
        # 추론용: batch_size(1), dype 수정요망
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cuda:0')
        # self.theta = torch.zeros(1, 2, 3, dtype=torch.float16, device='cuda:0')
        self.theta = torch.zeros(1, 2, 3, dtype=torch.float32, device='cpu')
        self.theta[:, 0, 0] = self.roi_size / 640
        self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

    def forward(self, x):
        self.ncxy = self.localizer(x)

        # 변환행렬 생성
        # self.theta = torch.zeros(x.size(0), 2, 3, dtype=x.dtype, device=x.device)
        self.theta[:, :, 2] = self.ncxy
        # self.theta[:, 0, 2] = self.ncxy[:, 0]  # tx, ty => [-1, 1]
        # self.theta[:, 1, 2] = self.ncxy[:, 1]
        # self.theta[:, 0, 0] = self.roi_size / 640
        # self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

        grid = F.affine_grid(self.theta, [x.size(0), 3, self.in_size, self.in_size], align_corners=False)
        self.roi = F.grid_sample(x, grid, align_corners=False)

        x = self.features(self.roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


class ClassifierNetVfOpt(nn.Module):
    def __init__(self, fp=32, mod='opt'):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, 4)

        if mod == 'opt':
            self.localizer = LocalizerNet3Dsconv()
        elif mod == 'best':
            self.localizer = LocalizerNetV2()
        elif mod == 'base':
            self.localizer = LocalizerNet()
        self.inv255 = 1.0 / 255.0

        # 추론용: batch_size(1), dype 수정요망
        dtype = torch.float16 if fp == 16 else torch.float32
        self.register_buffer('base_theta', torch.zeros(1, 2, 3, dtype=dtype))    # 모델과 함께 device 이동
        self.base_theta[:, 0, 0] = 64 / 640
        self.base_theta[:, 1, 1] = 64 / 480  # scale (x, y) = 1, rot = 0

    def forward(self, x):
        # 전처리 내재화: 입력 x가 uint8 (B, 3, H, W)라고 가정
        x = x.to(self.base_theta.dtype)     # FP16이면 FP16으로, FP32면 FP32로 캐스팅
        x = x.mul(self.inv255)

        # 변환행렬 생성
        theta = self.base_theta.expand(x.size(0), -1, -1).clone()
        theta[:, :, 2] = self.localizer(x)

        # Sampling
        grid = F.affine_grid(theta, [x.size(0), 3, 96, 96], align_corners=False)
        roi = F.grid_sample(x, grid, align_corners=False)

        x = self.features(roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (bs, 64)
        # x = self.drop(x)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)   # 추론에서는 삭제
        return x


class LocalizerNet3Dsconv(nn.Module):
    """RegModel34, Spatial Transformer Network -> LocalizerNet    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            dsconv_block(3, 16, 3, padding=1),        # 1/2 : (634, 474) -> (317, 237)
            dsconv_block(16, 32, 3, padding=1),       # 1/4 = (313, 233) -> (156, 116)
            dsconv_block(32, 48, 3, padding=1),       # 1/8 = (154, 114) -> (77, 57)
            conv_block(48, 60, 3, padding=1),       # 1/16 = (75, 55) -> (37, 27)
            conv_block(60, 72, 3, padding=1),       # 1/32 = (35, 25) -> (17, 12)
            conv_block(72, 84, 3, padding=1),       # 1/64 -> (15, 10) -> (7, 5)
            conv_block(84, 96, 3, padding=1)        # 1/128 -> (, 3)
        )
        self.fc = nn.Sequential(
            nn.Linear(1440, 64),    # W * H * C = 7 * 5 * 84 = 1440
            nn.Sigmoid(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


if __name__ == '__main__':
    # torch.save(model, 'clssifier.pth')
    
    # 샘플 이미지 만들기
    dummy_img = torch.randn((1, 3, 480, 640), dtype=torch.float32).to('cpu')
    # dummy_img = torch.randn((1, 3, 96, 96), dtype=torch.float32).to('cpu')

    # 모델 객체 생성
    acnn = ClassifierNetV2()
    # acnn = OnlyClassifierNet()
    acnn = acnn.localizer
    # acnn = LocalizerNet3Dsconv()

    # 모델 불러오기
    # print("\nLoading the ACNN model ...")
    # acnn_path = "/home/jetson/onnx2trt-env/2024_0926_0613_64_best_classifier.pth"     # ln_opt & cn
    # acnn = load_model(acnn_path, device='cpu', half=False)
    # base_path = "/home/jetson/onnx2trt-env/2024_0913_2055_base_localizer.pth"        # ln_base
    # base_ln = load_model(base_path, device=device, half=True)
    # best_path = "/home/jetson/onnx2trt-env/2024_0912_1544_best_localizer.pth"         # ln_best
    # best_ln = load_model(best_path, device=device, half=False)

    acnn.eval()
    # print(acnn.in_size)
       
    print(acnn)
    # summary(acnn, input_size=(3, 480, 640), device='cpu', batch_size=1)   # 오류발생
    # summary(acnn, input_size=(3, 96, 96), device='cpu')
    summary2(acnn)
    flops, params = profile(acnn.to('cuda:0'), inputs=(dummy_img.to('cuda:0'),), verbose=False)  # 연산량 계산
    print('Network : {:,.0f} Params, {:.2f} MFLOPs'.format(params, flops / 1e6))

    # 특정 레이어의 FLOPs와 파라미터 수 계산
    # macs = profile_macs(acnn.fc[0], torch.randn(1, 1440))
    # flops = 2 * macs
    # params = sum(p.numel() for p in acnn.fc[1].parameters())

    # 1단계 : 모델 분리 및 검증 -----------------------------------------------------------------
    # ln = acnn.localizer     # ln = LocalizerNet3Dsconv()
    # ln = base_ln.localizer
    # ln = best_ln.localizer
    # cn = ClassifierNetOnly(dropout=0)
    # cn = ClassifierNetOnly(dropout=0).half()
    # cn.features = acnn.features
    # cn.fc = acnn.fc

    pass
