""" Torch2trt 모듈(github)를 이용하여 opt model의 정확도 비교
     fp32 vs. fp 16
"""
import time
import sys
import os
import gc 
from glob import glob
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchsummary import summary
from torchinfo import summary as summary2
from thop import profile
from torch2trt import torch2trt
from torch2trt import TRTModule

from si02_model_ver9 import ClassifierNet, LocalizerNet, ClassifierNetV2, LocalizerNetV2, \
LocalizerNet3Dsconv, OnlyClassifierNet, ClassifierNetVf
# from acnn_info import ClassifierNetVf, LocalizerNet3Dsconv, ClassifierNetOnly
from myutils import load_model, convert_ndarray_to_tensor, load_label, track_time


# 지수 표기법 비활성화
torch.set_printoptions(sci_mode=False)

# 전역변수 설정
classnames = {0: 'p1069', 1: 'p1071', 2: 'p3707', 3: 'p3709'}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 입력 데이터
BATCH_SIZE = 1
USE_FP16 = True
USE_GC = False
print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else 'w/oGC')

# print("\nLoading the ACNN model ...")
# acnn_path = "/home/jetson/onnx2trt-env/2024_0926_0613_64_best_classifier.pth"     # ln_opt & cn
# acnn = load_model(acnn_path, device=device, half=USE_FP16)
# base_path = "/home/jetson/onnx2trt-env/2024_0913_2055_base_localizer.pth"        # ln_base
# base_ln = load_model(base_path, device=device, half=True)
# best_path = "/home/jetson/onnx2trt-env/2024_0912_1544_best_localizer.pth"         # ln_best
# best_ln = load_model(best_path, device=device, half=False)

"""# (옵션) 모델 정보 출력
# print(acnn)
# summary(acnn.to('cpu'), input_size=(3, 480, 640), device='cpu')
# summary2(acnn)
# flops, params = profile(acnn, inputs=(dummy_img,), verbose=False)  # 연산량 계산
# print('Network : {:,.0f} Params, {:.2f} MFLOPs'.format(params, flops / 1e6))

# 특정 레이어의 FLOPs와 파라미터 수 계산
# macs = profile_macs(acnn.fc[0], torch.randn(1, 1440))
# flops = 2 * macs
# params = sum(p.numel() for p in acnn.fc[1].parameters())
"""
"""# 1단계 : 모델 분리 (Torch 모델 검증용) -----------------------------------------------------------------
ln = acnn.localizer     # ln = LocalizerNet3Dsconv()
# ln = base_ln.localizer
# ln = best_ln.localizer
# cn = ClassifierNetOnly(dropout=0)
cn = ClassifierNetOnly(dropout=0).half()
cn.features = acnn.features
cn.fc = acnn.fc
"""

print('\nLoading the TensorRT model ...')
if USE_FP16:
    # ln_trt_path = '/home/jetson/onnx2trt-env/ln_trt_dict_fp16.pth';       print('acnn-opt-fp16')
    # ln_trt_path = '/home/jetson/onnx2trt-env/ln_best_trt_fp16.pth';     print('acnn-best-fp16')      
    ln_trt_path = '/home/jetson/onnx2trt-env/ln_base_trt_fp16.pth';     print('acnn-base-fp16')      
    cn_trt_path = '/home/jetson/onnx2trt-env/cn_trt_dict_fp16.pth'
else:
    # ln_trt_path = '/home/jetson/onnx2trt-env/ln_trt_dict.pth';                 print('acnn-opt-fp32')
    # ln_trt_path = '/home/jetson/onnx2trt-env/ln_best_trt_fp32.pth';     print('acnn-best-fp32')
    ln_trt_path = '/home/jetson/onnx2trt-env/ln_base_trt_fp32.pth';     print('acnn-base-fp32')
    cn_trt_path = '/home/jetson/onnx2trt-env/cn_trt_dict.pth'

# [TensorRT] WARNING: only a single logger pointer at any given time
ln_trt = TRTModule()
ln_trt.load_state_dict(torch.load(ln_trt_path))
cn_trt = TRTModule()
cn_trt.load_state_dict(torch.load(cn_trt_path))

"""레이블 가져오기
# img_paths = glob(os.path.join(img_folder, 'p1069*.jpg'))
# img_path = img_paths[1]
# lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
# gt_cls, gt_cx, gt_cy = load_label(lbl_path, resolution=None)
# print('\nData \n(img)', img_path, '\n(lbl)', lbl_path)
# print('(GT): (cx, cy) = ({}, {}), (class) {}'.format(gt_cx, gt_cy, classnames[gt_cls]))
"""


# 테스트 이미지 불러오기
img_folder = '/home/jetson/onnx2trt-env/datasets/images/test'
img_paths = glob(os.path.join(img_folder, '*.jpg'))
img_paths.sort(key=lambda x: os.path.basename(x))   # 파일 이름을 기준으로 정렬

# 추론용: batch_size(1), dype 수정요망
roi_size = 64
in_size = 96
if USE_FP16:
    theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float16, device='cuda:0')
else:
    theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float32, device='cuda:0')
theta[:, 0, 0] = roi_size / 640
theta[:, 1, 1] = roi_size / 480   


print('\nPredicting test images by TensorRT model ...')
num_errors = 0
for i, img_path in enumerate(img_paths):
    lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    gt_cls, gt_cx, gt_cy = load_label(lbl_path, resolution=None)
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:, :, ::-1] - 음수 stride 문제 발생
    img = torch.from_numpy(img).permute(2, 0, 1)
    
    img = img.cuda()    # 전송량 최소화 및 형변환/요소연산을 GPU에서 수행 
        
    if USE_FP16:
        img = img.half().div(255.0).unsqueeze(0)
    else:
        img = img.float().div(255.0).unsqueeze(0)
    
    theta[:, :, 2]  = ln_trt(img)
    
    grid = F.affine_grid(theta, [BATCH_SIZE, 3, in_size, in_size], align_corners=False)
    roi = F.grid_sample(img, grid, align_corners=False)
    
    out = cn_trt(roi)
    
    prob = torch.max(out) * 100
    cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
        
    print('{:3d} - {} ({:.1f}%)'.format(i+1, classnames[cls_id], prob), end='')
    # tx_ty = theta[0, :, 2].detach().float().cpu().numpy().copy()
    # print('     tx={:.4f}, ty={:.4f}'.format(tx_ty[0], tx_ty[1], end=''))
    # p4 = out[0].detach().float().cpu().numpy().copy()
    # print('     cls score=', p4, end='')

    if cls_id != gt_cls:
        num_errors += 1
        print(' <--- ', os.path.basename(img_path))
    print()
print('Error 개수 :', num_errors)

"""
mean_pre_time = np.array(pre_times).mean() 
mean_transfer_time = np.array(transfer_times).mean()
mean_forward_time = np.array(forward_times).mean()
mean_infer_time = np.array(infer_times).mean()
std_pre_time = np.array(pre_times).std() 
std_transfer_time = np.array(transfer_times).std()
std_forward_time = np.array(forward_times).std()
std_infer_time = np.array(infer_times).std()
ratio_pre_time = mean_pre_time / mean_infer_time * 100
ratio_transfer_time = mean_transfer_time / mean_infer_time * 100
ratio_forward_time = mean_forward_time / mean_infer_time * 100
print('Average preprocessing time ({:.2f}%) : {:.2f}ms +/- {:.2f} '.format(ratio_pre_time, mean_pre_time, std_pre_time))
print('Average transfer time ({:.2f}%) : {:.2f}ms +/- {:.2f} '.format(ratio_transfer_time, mean_transfer_time, std_transfer_time))
print('Average forward time ({:.2f}%) : {:.2f}ms +/- {:.2f}'.format(ratio_forward_time, mean_forward_time, std_forward_time))
print('-' * 50)
print('Average inference time : {:.2f}ms +/- {:.2f} '.format(mean_infer_time, std_infer_time))
"""
"""
infer_times = []
transfer_times = []
pre_times = []
forward_times = []
print('\nForwarding torch ACNN model ...')
for i in range(104):
    img = cv2.imread(img_paths[i % 4])
    t0 = time.time()
    
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    # img = torch.from_numpy(img).half().div(255.0).unsqueeze(0)
    t1 = time.time()
    
    img = img.to(device)
    t2 = time.time()
    
    out = acnn(img)
    t3 = time.time()
    
    prob = torch.max(out) * 100
    cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
    
    # 가비지 컬렉션 수행<-- 일정 수준이 넘어가면 추론시간이 +15ms
    gc.collect()
    
    pre_time = (t1- t0) * 1000			
    transfer_time = (t2-t1) * 1000
    forward_time = (t3 - t2) * 1000
    infer_time = (t3 - t0) * 1000
    if i > 3:
        pre_times.append(pre_time)
        transfer_times.append(transfer_time)
        forward_times.append(forward_time)
        infer_times.append(infer_time)
        infer_times.append(infer_time)
    else:
        print('{} - Inference time: {:.2f}ms, Output: {} ({:.1f}%)'.format(i+1, infer_time, classnames[cls_id], prob))

mean_pre_time = np.array(pre_times).mean() 
mean_transfer_time = np.array(transfer_times).mean()
mean_forward_time = np.array(forward_times).mean()
mean_infer_time = np.array(infer_times).mean()
std_pre_time = np.array(pre_times).std() 
std_transfer_time = np.array(transfer_times).std()
std_forward_time = np.array(forward_times).std()
std_infer_time = np.array(infer_times).std()
ratio_pre_time = mean_pre_time / mean_infer_time * 100
ratio_transfer_time = mean_transfer_time / mean_infer_time * 100
ratio_forward_time = mean_forward_time / mean_infer_time * 100
print('Average preprocessing time ({:.2f}%) : {:.2f}ms +/- {:.2f} '.format(ratio_pre_time, mean_pre_time, std_pre_time))
print('Average transfer time ({:.2f}%) : {:.2f}ms +/- {:.2f} '.format(ratio_transfer_time, mean_transfer_time, std_transfer_time))
print('Average forward time ({:.2f}%) : {:.2f}ms +/- {:.2f}'.format(ratio_forward_time, mean_forward_time, std_forward_time))
print('-' * 50)
print('Average inference time : {:.2f}ms +/- {:.2f} '.format(mean_infer_time, std_infer_time))

for infer_time in infer_times:
    print('{:10.2f}'.format(infer_time), end='')
"""


