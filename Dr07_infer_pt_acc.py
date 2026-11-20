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

from torch2trt import torch2trt
from torch2trt import TRTModule
from si02_model_ver9 import ClassifierNet, ClassifierNetV2, ClassifierNetVf, \
                                                    LocalizerNet3Dsconv, OnlyClassifierNet


# 지수 표기법 비활성화
torch.set_printoptions(sci_mode=False)

# 전역변수 설정
classnames = {0: 'p1069', 1: 'p1071', 2: 'p3707', 3: 'p3709'}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 입력 데이터
BATCH_SIZE = 1
USE_FP16 = True
USE_GC = False
INV255 = 1 / 255.0
# pin_mem = torch.zeros((480, 640, 3), dtype=torch.uint8).pin_memory()  # +1ms 
# print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else '')

# 샘플 이미지 만들기
# dummy_img = torch.zeros((BATCH_SIZE, 3, 480, 640), dtype=torch.float32, device=device)
# img_paths = [r'/home/jetson/onnx2trt-env/datasets/images/train/p1069_0000_04_v1f2.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/test/p1071_277_3.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/train/p3707_0195_10_v1f1.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/test/p3709_289_3.jpg']
# for i in range(BATCH_SIZE):
    # img = cv2.imread(img_paths[i % 4])
    # dummy_img[i] = convert_ndarray_to_tensor(img, bgr2rgb=True)

# 테스트 이미지 불러오기
img_folder = '/home/jetson/onnx2trt-env/datasets/images/test'
img_paths = glob(os.path.join(img_folder, '*.jpg'))
img_paths.sort(key=lambda x: os.path.basename(x))   # 파일 이름을 기준으로 정렬

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
acnn_path = "/home/jetson/onnx2trt-env/2024_0926_0613_64_best_classifier.pth"     # ln_opt & cn
acnn = load_model(acnn_path, device=device, half=USE_FP16)
# My model
# acnn =ClassifierNet(); print('ACNN-base model')			# base : 7x7, in : 96x96
# acnn = ClassifierNetV2();  print('ACNN-best model')	 # best : 32.78 +/- 6.6, size not supported
# model = OnlyClassifierNet()
# acnn = ClassifierNetVf(); 	acnn.in_size = 96;  print('ACNN-opt model') 	# opt
acnn.eval()
for param in acnn.parameters():
    param.requires_grad = False	
# print(acnn)

acnn.to(device)

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

print(device, '/ FP16' if USE_FP16 else '/ FP32', 'w/GC' if USE_GC else 'w/oGC')
print('\nForwarding model...')
gc.collect()
num_errors = 0
for i, img_path in enumerate(img_paths):
    lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    gt_cls, gt_cx, gt_cy = load_label(lbl_path, resolution=None)
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:, :, ::-1] - 음수 stride 문제 발생
    img = torch.from_numpy(img).permute(2, 0, 1)
    
    img = img.cuda()    # 전송량 최소화 및 형변환/요소연산을 GPU에서 수행 
        
    if USE_FP16:
        img = img.half().div(255.0).unsqueeze(0)   # div_(255.0), mul_(INV255) 차이없음
    else:
        img = img.float().div(255.0).unsqueeze(0)
    
    # forward model
    out = acnn(img)
    # theta[:, :, 2]  = acnn.localizer(img)       
    
    prob = torch.max(out) * 100
    cls_id = torch.argmax(out).item()   # 텐서를 스칼라로 변환
    
    print('{:3d} - {} ({:.1f}%)'.format(i+1, classnames[cls_id], prob), end='')
    tx_ty = acnn.theta[0, :, 2].detach().float().cpu().numpy().copy()
    print('     tx={:.4f}, ty={:.4f}'.format(tx_ty[0], tx_ty[1], end=''))
    p4 = out[0].detach().float().cpu().numpy().copy()
    print('     cls score=', p4, end='')
    
    if cls_id != gt_cls:
        num_errors += 1
        print(' <--- ', os.path.basename(img_path))
    print()
print('Error 개수 :', num_errors)
    
