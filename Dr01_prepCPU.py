import time
import torch
import cv2
import numpy as np
import gc


device = torch.device("cuda")
INV255 = 1 / 255.0
pin_img = torch.zeros((1, 3, 480, 640), dtype=torch.float32).pin_memory()
img_paths = [r'/home/jetson/onnx2trt-env/datasets/images/train/p1069_0000_04_v1f2.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p1071_277_3.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/train/p3707_0195_10_v1f1.jpg',
             r'/home/jetson/onnx2trt-env/datasets/images/test/p3709_289_3.jpg']
             
N =  34
n_warmup = 4
t = np.zeros((N, 8), dtype=np.float64) 
total = np.zeros(N, dtype=np.float64) 
med = np.zeros(8, dtype=np.float64)


for i in range(N):
    # 0. Load (CPU)
    img = cv2.imread(img_paths[i%4], cv2.IMREAD_COLOR)
    
    # 측정 시작
    gc.collect()
    torch.cuda.synchronize()
    t[i, 0] = time.perf_counter()
    
    ## Numpy 활용
    # 1. BGR to RGB (CPU)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 연속적인 메모리 배열을 반환
    # img = img[:, :, ::-1].copy()  # 5ms 정도 증가됨
    torch.cuda.synchronize()
    t[i, 1] = time.perf_counter()
    
    # 2. Numpy to Tensor (CPU)
    img = torch.from_numpy(img)
    torch.cuda.synchronize()
    t[i, 2] = time.perf_counter()
    
    # 3. To Float (CPU/GPU 전환점)
    img = img.float()      
    torch.cuda.synchronize()
    t[i, 3] = time.perf_counter()
    
    # 4. Normalize (GPU 연산일 경우 대비 동기화 필요)
    # img = img.div(255.0)    
    img = img.mul_(INV255)   # .div에 비해 0.5ms 단축
    torch.cuda.synchronize()
    t[i, 4] = time.perf_counter()
    
    # 명령어 chaining으로 메모리 캐 메모리 효율 향상, 메모리 할당 제거
    # img = img.permute(2, 0, 1).contiguous().float().mul_(INV255).unsqueeze(0)
    # img = img.permute(2, 0, 1).float().mul(INV255).unsqueeze(0)
    
    # 5. Permute & Contiguous (CPU -> GPU 명령 대기)
    # img = img.permute(2, 0, 1)    
    img = img.permute(2, 0, 1).contiguous()    # 추론이 안정적임
    # torch.cuda.synchronize()
    t[i, 5] = time.perf_counter()
    
    # 6. Batch Dimension
    img = img.unsqueeze(0)    
    torch.cuda.synchronize()
    t[i, 6] = time.perf_counter()
    
    # 7. Device Transfer (가장 핵심적인 부분)
    img = img.to(device)    # 0.2ms 빠르고, 형변환도 동시에 가능
    # img = img.cuda()
    torch.cuda.synchronize()    # GPU 작업이 끝날 때까지 대기함 
    t[i, 7] = time.perf_counter()
    
    # 메모리 관리 (측정 루프 내 gc는 가급적 피하는 것이 좋으나 필요시 유지)
    # gc.collect()

    
print("\n --- Preprocessing Time (seconds) ---")
print("Index         RGB   Tensor   Dtype   Norm    CHW     Batch   Trans    Total")
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

print("  Ratio   : ", end = ' ')
for j in range(8): 
    ratio = med[j] / med[7] * 100
    print(f'{ratio:4.2f}%', end='  ')
print("\n", "-" * 75)
print("Index         RGB   Tensor   Dtype   Norm    CHW     Batch   Trans    Total")
