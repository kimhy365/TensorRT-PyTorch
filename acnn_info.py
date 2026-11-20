# ========================================================================
# ONNX 변환을 위한 A-CNN 모델의 정보
# ========================================================================
# import sys
import warnings
warnings.filterwarnings('ignore')

import random
import torch.nn.functional as F
from myutils import *


# 난수 시드 설정
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)

# 전역변수 설정
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Models ################################################################
# Version 1 : base
class LocalizerNet(nn.Module):
    """RegModel34, Spatial Transformer Network -> LocalizerNet    """
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
        self.fc = nn.Sequential(
            nn.Linear(1440, 64),    # W * H * C = 5 * 3 * 96 = 1440
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


# Version 2 : best - 3x3, sigmoid
class LocalizerNetV2(nn.Module):
    def __init__(self, _dropout=0):
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

        self.fc = nn.Sequential(
            nn.Linear(1440, 64),    # W * H * C = 5 * 3 * 96 = 1440
            nn.Sigmoid(),   # 20240722
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


# Version 3 : opt - best & Dsconv
class LocalizerNet3Dsconv(nn.Module):
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (bs, 96, 3, 5)
        x = self.fc(x)
        return x        # (bs, ncx, ncy)


''' affine_gird(), grid_sample() 함수 대체방법 --> 실패
# https://zenn.dev/pinto0309/scraps/63e5d87b497641
def affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    grid = create_grid(N, C, H, W, align_corners)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid


def create_grid(N, C, H, W, align_corners):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, align_corners))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, align_corners).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid


def linspace_from_neg_one(num_steps, align_corners, dtype=torch.float32):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
    if not align_corners:
        r = r * (num_steps - 1) / num_steps
    return r


def patch_affine_grid_generator():
    torch.nn.functional.affine_grid = affine_grid


def grid_sample1(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

'''


class ClassifierNetVf(nn.Module):
    def __init__(self, roi_size=64, in_size=64, c_dropout=0.0):
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
        self.roi = None
        self.roi_size = roi_size
        self.in_size = in_size
        self.ncxy = None
        
        # 추론용: batch_size(1), dype 수정요망
        self.theta = torch.zeros(1, 2, 3, dtype=torch.float16, device='cuda:0')
        self.theta[:, 0, 0] = self.roi_size / 640
        self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

    def forward_통합테스트(self, x): 	# For inference
        self.ncxy = self.localizer(x)

        # # 중심좌표 계산 및 roi 추출
        # cx = int((self.ncxy[:, 0] + 1) * 320)
        # cy = int((self.ncxy[:, 1] + 1) * 240)
        # x1 = min(max(cx - 32, 0), 576)     # 576 = W - 64
        # y1 = min(max(cy - 32, 0), 416)     # 416 = H - 64
        # x2 = x1 + 64
        # y2 = y1 + 64
        # self.roi = F.interpolate(x[:, :, y1:y2, x1:x2], size=(96, 96), mode='bilinear', align_corners=False)
        
        # 중심 좌표 및 ROI 좌표를 한 번에 계산
        cxy = ((self.ncxy + 1) * torch.tensor([320, 240], dtype=x.dtype, device=x.device)).int()
        xy1 = torch.clamp(cxy - 32, min=0, max=torch.tensor([576, 416], dtype=x.dtype, device=x.device))  # 576=W-64, 416=H-64
        xy2 = xy1 + 64

        # ROI 추출 (배치 처리)
        roi = F.interpolate(x[:, :, xy1[:, 1]:xy2[:, 1], xy1[:, 0]:xy2[:, 0]], size=(96, 96), mode='bilinear', align_corners=False)

        x = self.features(self.roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x): 	# for training
        self.ncxy = self.localizer(x)

        # 변환행렬 생성
        self.theta = torch.zeros(x.size(0), 2, 3, dtype=x.dtype, device=x.device)
        self.theta[:, 0, 2] = self.ncxy[:, 0]  # tx, ty => [-1, 1]
        self.theta[:, 1, 2] = self.ncxy[:, 1]
        # self.theta[:, :, 2] = self.ncxy
        self.theta[:, 0, 0] = self.roi_size / 640
        self.theta[:, 1, 1] = self.roi_size / 480  # scale (x, y) = 1, rot = 0

        grid = F.affine_grid(self.theta, [x.size(0), 3, self.in_size, self.in_size], align_corners=False)
        # grid = my_affine_grid(self.theta, [x.size(0), 3, self.in_size, self.in_size], align_corners=False)
        self.roi = F.grid_sample(x, grid, align_corners=False)
        # self.roi = my_grid_sample(x, grid, align_corners=False)

        x = self.features(self.roi)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


class ClassifierNetOnly(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3, padding=1),  # /2
            conv_block(16, 32, 3, padding=1),  # /4
            conv_block(32, 64, 3, padding=1),  # /8
            conv_block(64, 128, 3, padding=1),  # /16
            conv_block(128, 256, 3, padding=1),  # /32 -> 2*2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # (bs, 64)
        x = self.drop(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def conv_block(num_in, num_out, kernel_size, padding=0, stride=1):
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


# affine_grid 대체함수
def my_affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    grid = create_grid(N, C, H, W, align_corners)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid


def create_grid(N, C, H, W, align_corners):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, align_corners))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, align_corners).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid


def linspace_from_neg_one(num_steps, align_corners, dtype=torch.float32):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
    if not align_corners:
        r = r * (num_steps - 1) / num_steps
    return r


# grid_sample() 대체함수
def my_grid_sample(im, grid, align_corners=False):
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn
    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2
    x = x.view(n, -1)
    y = y.view(n, -1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)
    im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)
    im_padded = im_padded.view(n, c, -1)
    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)
    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)
