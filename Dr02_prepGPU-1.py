import time
import torch
import cv2
import numpy as np
import gc
from glob import glob
# import random


device = torch.device("cuda")
INV255 = 1 / 255.0
N = 2000
n_warmup = 0
# pin_img = torch.zeros((3, 480, 640), dtype=torch.uint8).pin_memory()
img_paths = glob(r'/home/jetson/onnx2trt-env/datasets/images/train/*.jpg')
# img_paths = random.sample(img_paths, N)
# img_paths = [r'/home/jetson/onnx2trt-env/datasets/images/train/p1069_0000_04_v1f2.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/test/p1071_277_3.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/train/p3707_0195_10_v1f1.jpg',
             # r'/home/jetson/onnx2trt-env/datasets/images/train/p3709_0189_17_v2f3.jpg']
img_paths = img_paths[:N+n_warmup]
n_img = len(img_paths)
print(n_img)
             
t = np.zeros((N, 8), dtype=np.float64) 
total = np.zeros(N, dtype=np.float64) 
med = np.zeros(8, dtype=np.float64)
std = np.zeros(8, dtype=np.float64)
torch.cuda.empty_cache()

for i in range(N):
    # 0. Load (CPU)
    img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
    
    # 측정 시작
    # gc.collect()      # 일종의 외란
    torch.cuda.synchronize()    # GPU 작업이 끝날 때까지 대기함 
    t[i, 0] = time.perf_counter()
    
    # 1. BGR to RGB (CPU)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 연속적인 메모리 배열을 반환
    torch.cuda.synchronize()
    t[i, 1] = time.perf_counter()
    
    # 2. Numpy to Tensor (CPU)
    img = torch.from_numpy(img)
    torch.cuda.synchronize()
    t[i, 2] = time.perf_counter()
    
    # 3. Permute & Contiguous (CPU -> GPU 명령 대기)
    img = img.permute(2, 0, 1)
    torch.cuda.synchronize()    # GPU 작업이 끝날 때까지 대기함 
    t[i, 3] = time.perf_counter()
    
    # 6. Batch Dimension
    img = img.unsqueeze(0)    
    torch.cuda.synchronize() 
    t[i, 4] = time.perf_counter()   
    
    # 7. Device Transfer (가장 핵심적인 부분)
    img = img.to(device)    # img.cuda()
    torch.cuda.synchronize() 
    t[i, 5] = time.perf_counter()
    
    # 4. To Float (CPU/GPU 전환점)
    img = img.float()
    torch.cuda.synchronize() 
    t[i, 6] = time.perf_counter()
    
    # 5. Normalize (GPU 연산일 경우 대비 동기화 필요)
    img = img.mul_(INV255)    # _이 0.01ms 빠름 vs. mul, div_, div
    torch.cuda.synchronize() 
    t[i, 7] = time.perf_counter()
        
    
print("\n --- Preprocessing Time (seconds) ---")
print("Index         RGB   Tensor   CHW    Batch   Trans  Dtype      Norm   Total")
print("-" * 75)
for i in range(n_warmup, N):
    print(f'Index {i-n_warmup+1:03d} :', end='  ')
    for j in range(7): 
        dt = (t[i, j+1] - t[i, j]) * 1000
        print(f'{dt:.3f}', end='   ')
    total[i] = (t[i, 7] - t[i, 0]) * 1000 
    print(f'{total[i]:.3f}')
    
print("-" * 75)
print("  Median  : ", end = ' ')
for j in range(7): 
    med[j] = np.median((t[n_warmup:, j+1] - t[n_warmup:, j]) * 1000)
    print(f'{med[j]:.3f}', end='   ')
med[7] = np.sum(med[:-1])
print(f'{med[-1]:.3f}')

print("  Percent : ", end = ' ')
for j in range(8): 
    ratio = med[j] / med[7] * 100
    print(f'{ratio:5.1f}%', end='  ')
    
print("\n  Stdev   : ", end = ' ')
for j in range(7): 
    std[j] = np.std((t[n_warmup:, j+1] - t[n_warmup:, j]) * 1000)
    print(f'{std[j]:.3f}', end='   ')
std[7] = np.mean(std[:-1])
print(f'{std[-1]:.3f}')
    
print("\n", "-" * 75)
print("Index         RGB   Tensor   CHW    Batch   Trans  Dtype      Norm   Total")
