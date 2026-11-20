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
from myutils import load_model, convert_ndarray_to_tensor, convert_tensor_image_to_opencv
from acnn_info import ClassifierNetVf, LocalizerNet3Dsconv, ClassifierNetOnly
import torch
import gc 

from torch2trt import torch2trt
from torch2trt import TRTModule
# from p8_cn_train import ClassifierNetVf, LocalizerNet3Dsconv
from acnn_info import ClassifierNetVf, LocalizerNet3Dsconv

"""
print('현재 작업 디렉토리 :', os.getcwd())
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
print('현재 작업 디렉토리 :', os.getcwd())
sys.exit()"""
os.chdir('../torch2trt-env')

# 지수 표기법 비활성화
torch.set_printoptions(sci_mode=False)

# 전역변수 설정
classnames = {0: 'p1069', 1: 'p1071', 2: 'p3707', 3: 'p3709'}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device, ' # CPU에서는 모델의 결과가 nan으로 나타남')

# 입력 데이터
BATCH_SIZE = 1

# ONNX 변환시
# dummy_img = torch.rand(BATCH_SIZE, 3, 480, 640).to(device)      # randn이나 별 차이가 없음
dummy_img = torch.rand(BATCH_SIZE, 3, 480, 640).half().to(device)      # randn이나 별 차이가 없음
# dummy_img = torch.rand(BATCH_SIZE, 3, 96, 96).to(device)      # randn이나 별 차이가 없음
# dummy_img = torch.rand(BATCH_SIZE, 3, 96, 96).half().to(device)      # randn이나 별 차이가 없음

# (옵션) 검증시 : 실제 이미지로 dummy_img 만들기
# dummy_img = torch.zeros((BATCH_SIZE, 3, 480, 640), dtype=torch.float32, device=device)
img_paths = [r'/home/jetson/onnx2trt-env/datasets/images/train/p1069_0000_04_v1f2.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p1071_277_3.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/train/p3707_0195_10_v1f1.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p3709_289_3.jpg']
# for i in range(BATCH_SIZE):
    # img = cv2.imread(img_paths[i % 4])
    # dummy_img[i] = convert_ndarray_to_tensor(img, bgr2rgb=True)

# load the pretrained model
model_path = "2024_0926_0613_64_best_classifier.pth"
acnn = load_model(model_path, device=device, half=True)      # "cuda:0", eval & weight_fixed
# out = acnn(dummy_img)
# print('\nA-CNN : \n(LN)', acnn.ncxy, '\n(CN)', out)

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
ln = acnn.localizer     # ln = LocalizerNet3Dsconv()
# cn = ClassifierNetOnly(dropout=0).half()
# cn.features = acnn.features
# cn.fc = acnn.fc

""" 
# TensorRT 변환
print('\nConverting torch to tensorrt...')
model_trt = torch2trt(ln, [dummy_img], fp16_mode=True)

print('\nSaving the trt model...')
torch.save(model_trt.state_dict(), 'ln_trt_dict_fp16.pth') 
sys.exit()
""" 

print('\nLoading the trt model...')
model_trt = TRTModule()
model_trt_path = '/home/jetson/onnx2trt-env/ln_trt_dict_fp16.pth'
model_trt_path = '/home/jetson/onnx2trt-env/cn_trt_dict_fp16.pth'
model_trt.load_state_dict(torch.load(model_trt_path))

print('\nForwarding torch model...')
infer_times = []
for i in range(105):
    t0 = time.time()
    # out = ln(dummy_img)
    out = cn(dummy_img)
    infer_time = (time.time() - t0) * 1000    
    if i > 4:
        infer_times.append(infer_time)
    else:
        print('Inference time: {:.2f}ms, Output :'.format(infer_time), out)
        
    # 가비지 컬렉션 수행<-- 일정 수준이 넘어가면 추론시간이 +15ms
    gc.collect()

print('Average infer time: {:.2f}ms +-{:.2f}'.format(np.array(infer_times).mean(), np.array(infer_times).std()))
for infer_time in infer_times:
    print('{:10.2f}'.format(infer_time), end='')

print('\n\nForwarding tensorrt model...')
infer_times = []
for i in range(105):
    t0 = time.time()
    out = model_trt(dummy_img)
    infer_time = (time.time() - t0) * 1000

    if i > 4:
        infer_times.append(infer_time)
    else:
        print('Inference time: {:.2f}ms, Output :'.format(infer_time), out)
        
    gc.collect()

print('Average infer time: {:.2f}ms +-{:.2f}'.format(np.array(infer_times).mean(), np.array(infer_times).std()))
for infer_time in infer_times:
    print('{:10.2f}'.format(infer_time), end='')

