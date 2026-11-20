""" Torch2trt 모듈(github)를 이용하여 추론시간 비교
ACNN --> LN, CN(2)
Precision : FP32, FP16 - 입력 데이터, 모델 .half()
Framework : Torch, TensorRT 
"""
import torch
import time
import sys
import torch.nn.functional as F
import cv2
import os
from torchsummary import summary
from torchinfo import summary as summary2
from thop import profile
import numpy as np
from myutils import load_model, convert_ndarray_to_tensor, load_label, track_time
import torch
import gc 
import warnings

from torch2trt import torch2trt
from torch2trt import TRTModule
from Dr09_model_info import ClassifierNet, ClassifierNetV2, ClassifierNetVf, \
                                                    LocalizerNet3Dsconv, OnlyClassifierNet


# 지수 표기법 비활성화
torch.set_printoptions(sci_mode=False)

# 전역변수 설정
classnames = {0: 'p1069', 1: 'p1071', 2: 'p3707', 3: 'p3709'}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 입력 데이터
BATCH_SIZE = 1
USE_FP16 = False
USE_GC = False
INV255 = 1 / 255.0
# pin_mem = torch.zeros((480, 640, 3), dtype=torch.uint8).pin_memory()  # +1ms 
# print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else '')

# 샘플 이미지 만들기
# dummy_img = torch.zeros((BATCH_SIZE, 3, 480, 640), dtype=torch.float32, device=device)
img_paths = [r'/home/jetson/onnx2trt-env/datasets/images/train/p1069_0000_04_v1f2.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p1071_277_3.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/train/p3707_0195_10_v1f1.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p3709_289_3.jpg']
# for i in range(BATCH_SIZE):
    # img = cv2.imread(img_paths[i % 4])
    # dummy_img[i] = convert_ndarray_to_tensor(img, bgr2rgb=True)

"""
# print("\nLoading the ACNN model ...")
# acnn_path = "/home/jetson/onnx2trt-env/2024_0926_0613_64_best_classifier.pth"     # ln_opt & cn
# acnn = load_model(acnn_path, device=device, half=False)
# base_path = "/home/jetson/onnx2trt-env/2024_0913_2055_base_localizer.pth"        # ln_base
# base_ln = load_model(base_path, device=device, half=True)
best_path = "/home/jetson/onnx2trt-env/2024_0912_1544_best_localizer.pth"         # ln_best
best_ln = load_model(best_path, device=device, half=False)

# (옵션) 모델 정보 출력
# print(acnn)
# summary(acnn.to('cpu'), input_size=(3, 480, 640), device='cpu')
# summary2(acnn)
# flops, params = profile(acnn, inputs=(dummy_img,), verbose=False)  # 연산량 계산
# print('Network : {:,.0f} Params, {:.2f} MFLOPs'.format(params, flops / 1e6))

# 특정 레이어의 FLOPs와 파라미터 수 계산
# macs = profile_macs(acnn.fc[0], torch.randn(1, 1440))
# flops = 2 * macs
# params = sum(p.numel() for p in acnn.fc[1].parameters())

# 1단계 : 모델 분리 및 검증 -----------------------------------------------------------------
# ln = acnn.localizer     # ln = LocalizerNet3Dsconv()
# ln = base_ln.localizer
ln = best_ln.localizer
# cn = ClassifierNetOnly(dropout=0)
# cn = ClassifierNetOnly(dropout=0).half()
# cn.features = acnn.features
# cn.fc = acnn.fc

# TensorRT 변환
print('\nConverting torch to tensorrt...')
model_trt = torch2trt(ln, [dummy_img], fp16_mode=False)

print('\nSaving the trt model...')
torch.save(model_trt.state_dict(), 'ln_best_trt_fp32.pth') 
sys.exit()
"""

print('\nLoading the PyTorch ', end='')
# My model
acnn =ClassifierNet(); print('ACNN-base model')			# base : 7x7, in : 96x96
# acnn = ClassifierNetV2();  print('ACNN-best model')	 # best : 32.78 +/- 6.6, size not supported
# model = OnlyClassifierNet()
# acnn = ClassifierNetVf(); 	acnn.in_size = 96;  print('ACNN-opt model') 	# opt
acnn.eval()
for param in acnn.parameters():
    param.requires_grad = False	
# print(acnn)

acnn.to(device)

"""
# img_paths = glob(os.path.join(img_folder, 'p1069*.jpg'))
# img_path = img_paths[1]
# lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
# gt_cls, gt_cx, gt_cy = load_label(lbl_path, resolution=None)
# print('\nData \n(img)', img_path, '\n(lbl)', lbl_path)
# print('(GT): (cx, cy) = ({}, {}), (class) {}'.format(gt_cx, gt_cy, classnames[gt_cls]))
"""

# 추론용: batch_size(1), dype 수정요망
roi_size = 64
in_size = 96
if USE_FP16:
    acnn.half()
    acnn.theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float16, device='cuda:0')
else:
    acnn.theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float32, device='cuda:0')
acnn.theta[:, 0, 0] = roi_size / 640
acnn.theta[:, 1, 1] = roi_size / 480   

N = 1004
infer_times = np.zeros(N, dtype=np.float16)
transfer_times = np.zeros(N, dtype=np.float16)
pre_times = np.zeros(N, dtype=np.float16)
forward_times = np.zeros((3, N), dtype=np.float16)

print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else 'w/oGC')
print('\nForwarding model...')
gc.collect()
for i in range(N):
    img = cv2.imread(img_paths[i % 4], cv2.IMREAD_COLOR)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:, :, ::-1] - 음수 stride 문제 발생
    img = torch.from_numpy(img)
    # img = pin_mem.copy_(torch.from_numpy(img))    # +1ms
    # torch.cuda.synchronize()
    # t1 = time.perf_counter()    
    
    img = img.to(device)    # 전송량 최소화 및 형변환/요소연산을 GPU에서 수행 
    # img = img.to(device, non_blocking=True)    # pinned memory : +1ms 
    # torch.cuda.synchronize()
    # t2 = time.perf_counter()
        
    if USE_FP16:
        img = img.half().div(255.0).permute(2, 0, 1).unsqueeze(0)   # div_(255.0), mul_(INV255) 차이없음
    else:
        img = img.float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    # torch.cuda.synchronize()
    # t3 = time.perf_counter()
    
    # forward model
    out = acnn(img)
    # cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
    torch.cuda.synchronize()
    t4 = time.perf_counter()    
    
    # theta[:, :, 2]  = acnn.localizer(img)       
    # torch.cuda.synchronize()
    # t4_1 = time.perf_counter()
    
    # grid = F.affine_grid(theta, [BATCH_SIZE, 3, in_size, in_size], align_corners=False)
    # roi = F.grid_sample(img, grid, align_corners=False)
    # torch.cuda.synchronize()
    # t4_2 = time.perf_counter()
    
    # out = acnn.features(roi)
    # out = acnn.avgpool(out)
    # out = out.view(out.size(0), -1)       # (bs, 64)
    # out = acnn.fc(out)
    # out = F.softmax(out, dim=1)
    # torch.cuda.synchronize()
    # t4 = time.perf_counter()
    
    prob = torch.max(out) * 100
    cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
    
    # 가비지 컬렉션 수행<-- 일정 수준이 넘어가면 추론시간이 +15ms
    if USE_GC:
        gc.collect()
    
    # pre_times[i] = (t1 - t0 + t3 - t2) * 1000    
    # transfer_times[i] = (t2 - t1) * 1000
    # forward_times[0, i] = (t4_1 - t3) * 1000    
    # forward_times[1, i] = (t4_2 - t4_1) * 1000
    # forward_times[2, i] = (t4 - t4_2) * 1000
    infer_times[i] = (t4 - t0) * 1000    
    if i < 4:
        print(' {}-Inference time: {:.2f}ms, Output: {} ({:.1f}%)'.format(i+1, infer_times[i], classnames[cls_id], prob))

# ratio_pre = np.median(pre_times[4:]) / np.median(infer_times[4:]) * 100
# ratio_transfer = np.median(transfer_times[4:]) / np.median(infer_times[4:]) * 100
# ratio_forward_1 = np.median(forward_times[0, 4:]) / np.median(infer_times[4:]) * 100
# ratio_forward_2 = np.median(forward_times[1, 4:]) / np.median(infer_times[4:]) * 100
# ratio_forward_3 = np.median(forward_times[2, 4:]) / np.median(infer_times[4:]) * 100
print('-' * 70)
# print('Preprocess ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_pre, \
# np.median(pre_times[4:]) , pre_times[4:].mean(), np.percentile(pre_times[4:], 95), np.percentile(pre_times[4:], 99)))
# print('Transfer ({:.2f}%)  : med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_transfer, \
# np.median(transfer_times[4:]), transfer_times[4:].mean(), np.percentile(transfer_times[4:], 95), np.percentile(transfer_times[4:], 99)))
# print('Forward_1 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_1, \
# np.median(forward_times[0, 4:]), forward_times[0, 4:].mean(), np.percentile(forward_times[0, 4:], 95), np.percentile(forward_times[0, 4:], 99)))
# print('Forward_2 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_2, \
# np.median(forward_times[1, 4:]), forward_times[1, 4:].mean(), np.percentile(forward_times[1, 4:], 95), np.percentile(forward_times[1, 4:], 99)))
# print('Forward_3 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_3, \
# np.median(forward_times[2, 4:]), forward_times[2, 4:].mean(), np.percentile(forward_times[2, 4:], 95), np.percentile(forward_times[2, 4:], 99)))
# print('-' * 70)
print('Inference time    : med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format( \
np.median(infer_times[4:]), infer_times[4:].mean(), np.percentile(infer_times[4:], 95), np.percentile(infer_times[4:], 99)))
print('-' * 70)
for infer_time in infer_times[4:]:
    print('{:8.2f}'.format(infer_time), end='')


# 자식 객체(Module)부터 명시적으로 삭제
if 'cn_trt' in globals(): del cn_trt
if 'ln_trt' in globals(): del ln_trt

# GPU 메모리 캐시 정리
torch.cuda.empty_cache()
gc.collect()
