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
from glob import glob

# from torch2trt import torch2trt
# from torch2trt import TRTModule
from Dr09_model_info import ClassifierNet, ClassifierNetV2, ClassifierNetVf, \
                                                    LocalizerNet3Dsconv, OnlyClassifierNet


# 지수 표기법 비활성화
torch.set_printoptions(sci_mode=False)

# 전역변수 설정
classnames = {0: 'p1069', 1: 'p1071', 2: 'p3707', 3: 'p3709'}
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'      # not implemented for Half_CPU

# 입력 데이터
BATCH_SIZE = 1
USE_FP16 = False
USE_GC = False
INV255 = 1 / 255.0
# print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else '')

# 샘플 이미지 만들기
img_paths = glob(r'/home/jetson/onnx2trt-env/datasets/images/train/*.jpg')

print('\nLoading the PyTorch ', end='')
# My model
acnn =ClassifierNet(); print('ACNN-base model')			# base : 7x7, in : 96x96
# acnn = ClassifierNetV2();  print('ACNN-best model')	 # best : 32.78 +/- 6.6, size not supported
# model = OnlyClassifierNet()
# acnn = ClassifierNetVf(); 	acnn.in_size = 96;  print('ACNN-opt model') 	# opt
acnn.eval()
for param in acnn.parameters():
    param.requires_grad = False	


# 추론용: batch_size(1), dype 수정요망
roi_size = 64
in_size = 96
if USE_FP16:
    acnn.half()
    theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float16, device=device)
else:
    theta = torch.zeros(BATCH_SIZE, 2, 3, dtype=torch.float32, device=device)
theta[:, 0, 0] = roi_size / 640
theta[:, 1, 1] = roi_size / 480   

N = 204
n_warmup = 4
infer_times = np.zeros(N, dtype=np.float64)
transfer_times = np.zeros(N, dtype=np.float64)
pre_times = np.zeros(N, dtype=np.float64)
forward_times = np.zeros((3, N), dtype=np.float64)
post_times = np.zeros(N, dtype=np.float64)


img_paths = img_paths[:N+n_warmup]
n_img = len(img_paths)
print(n_img)

print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else 'w/oGC')
print('\nForwarding model...')
gc.collect()
for i in range(N):
    img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
    
    t0 = time.perf_counter()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:, :, ::-1] - 음수 stride 문제 발생
    img = torch.from_numpy(img)
    t1 = time.perf_counter()    
    
    # img = img.to(device)    # 전송량 최소화 및 형변환/요소연산을 GPU에서 수행 
    t2 = time.perf_counter()
        
    if USE_FP16:
        img = img.half().mul_(INV255).permute(2, 0, 1).unsqueeze(0)   # div_(255.0), mul_(INV255) 차이없음
    else:
        img = img.float().mul_(INV255).permute(2, 0, 1).unsqueeze(0)
    t3 = time.perf_counter()
    
    # forwarding the whole model
    # out = acnn(img)
    # torch.cuda.synchronize()
    # t3 = time.perf_counter()    
    
    theta[:, :, 2]  = acnn.localizer(img)       
    t4_1 = time.perf_counter()
    
    grid = F.affine_grid(theta, [BATCH_SIZE, 3, in_size, in_size], align_corners=False)
    roi = F.grid_sample(img, grid, align_corners=False)
    t4_2 = time.perf_counter()
    
    out = acnn.features(roi)
    out = acnn.avgpool(out)
    out = out.view(out.size(0), -1)       # (bs, 64)
    out = acnn.fc(out)
    out = F.softmax(out, dim=1)
    t4 = time.perf_counter()
    
    cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
    prob = torch.max(out) * 100
    t5 = time.perf_counter()

    
    # 가비지 컬렉션 수행<-- 일정 수준이 넘어가면 추론시간이 +15ms
    if USE_GC:
        gc.collect()
    
    pre_times[i] = (t1 - t0 + t3 - t2) * 1000    
    # pre_times[i] = (t1 - t0) * 1000    
    transfer_times[i] = (t2 - t1) * 1000
    forward_times[0, i] = (t4_1 - t3) * 1000    
    forward_times[1, i] = (t4_2 - t4_1) * 1000
    forward_times[2, i] = (t4 - t4_2) * 1000
    infer_times[i] = (t4 - t0) * 1000    
    post_times[i] = (t5 - t4) * 1000
    if i < 4:
        print(' {}-Inference time: {:.2f}ms, Output: {} ({:.1f}%)'.format(i+1, infer_times[i], classnames[cls_id], prob))

ratio_pre = np.median(pre_times[4:]) / np.median(infer_times[4:]) * 100
ratio_transfer = np.median(transfer_times[4:]) / np.median(infer_times[4:]) * 100
ratio_forward_1 = np.median(forward_times[0, 4:]) / np.median(infer_times[4:]) * 100
ratio_forward_2 = np.median(forward_times[1, 4:]) / np.median(infer_times[4:]) * 100
ratio_forward_3 = np.median(forward_times[2, 4:]) / np.median(infer_times[4:]) * 100
ratio_post = np.median(post_times[4:]) / np.median(infer_times[4:]) * 100
print('-' * 70)
print('Preprocess ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_pre, \
np.median(pre_times[4:]) , pre_times[4:].mean(), np.percentile(pre_times[4:], 95), np.percentile(pre_times[4:], 99)))
print('Transfer ({:.2f}%)  : med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_transfer, \
np.median(transfer_times[4:]), transfer_times[4:].mean(), np.percentile(transfer_times[4:], 95), np.percentile(transfer_times[4:], 99)))
print('Forward_1 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_1, \
np.median(forward_times[0, 4:]), forward_times[0, 4:].mean(), np.percentile(forward_times[0, 4:], 95), np.percentile(forward_times[0, 4:], 99)))
print('Forward_2 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_2, \
np.median(forward_times[1, 4:]), forward_times[1, 4:].mean(), np.percentile(forward_times[1, 4:], 95), np.percentile(forward_times[1, 4:], 99)))
print('Forward_3 ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_forward_3, \
np.median(forward_times[2, 4:]), forward_times[2, 4:].mean(), np.percentile(forward_times[2, 4:], 95), np.percentile(forward_times[2, 4:], 99)))
print('Postprocess ({:.2f}%): med/avg={:.2f}/{:.2f}ms, P95={:.2f}ms, P99={:.2f}ms '.format(ratio_post, \
np.median(post_times[4:]) , post_times[4:].mean(), np.percentile(post_times[4:], 95), np.percentile(post_times[4:], 99)))
print('-' * 70)
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
