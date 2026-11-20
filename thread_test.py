"""Thread test for cv2.cvtColor() in the two preprocessing pipelines (Table 3, Section 4.1.1).

Steps and their order are taken from the original measurement scripts:
  --pipe cpu : Dr01_prepCPU.py  (RGB, Tensor, Dtype, Norm, CHW(contiguous), Batch, Trans)
  --pipe gpu : Dr02_prepGPU.py  (RGB, Tensor, Trans, Dtype, Norm, CHW, Batch)
Both pipelines use the same image list, so only the pipeline and the thread setting differ.

Usage:
  python3 thread_test.py --pipe cpu|gpu [--cv N] [--torch N] [--half]
    --cv N    : cv2.setNumThreads(N)      (default: not called)
    --torch N : torch.set_num_threads(N)  (default: not called)
"""
import os
import time
import argparse
from glob import glob

import numpy as np
import torch
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--pipe', choices=['cpu', 'gpu'], required=True)
ap.add_argument('--cv', type=int, default=None)
ap.add_argument('--torch', type=int, default=None)
ap.add_argument('--half', action='store_true', help='use .half() instead of .float()')
ap.add_argument('--n', type=int, default=1000, help='measured runs')
ap.add_argument('--warmup', type=int, default=500, help='runs discarded as warm-up')
ap.add_argument('--img_dir', default='/home/jetson/onnx2trt-env/datasets/images/train')
a = ap.parse_args()

if a.cv is not None:
    cv2.setNumThreads(a.cv)
if a.torch is not None:
    torch.set_num_threads(a.torch)

par = [l.strip() for l in cv2.getBuildInformation().splitlines() if 'Parallel framework' in l]
print(f'# pipe={a.pipe} half={a.half} | cv2.getNumThreads()={cv2.getNumThreads()} '
      f'torch.get_num_threads()={torch.get_num_threads()} | {par} | '
      f'OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")} '
      f'OMP_WAIT_POLICY={os.environ.get("OMP_WAIT_POLICY")}')

device = torch.device('cuda')
INV255 = 1 / 255.0
paths = sorted(glob(os.path.join(a.img_dir, '*.jpg')))[:a.n + a.warmup]
N = len(paths)
t = np.zeros((N, 8), dtype=np.float64)
_ = torch.zeros(1, device=device)           # CUDA context initialization
torch.cuda.synchronize()


def to_dtype(x):
    return x.half() if a.half else x.float()


for i, p in enumerate(paths):
    img = cv2.imread(p, cv2.IMREAD_COLOR)   # image load is excluded, as in the paper

    torch.cuda.synchronize(); t[i, 0] = time.perf_counter()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                     # 1. RGB (CPU)
    torch.cuda.synchronize(); t[i, 1] = time.perf_counter()
    img = torch.from_numpy(img)                                     # 2. Tensor
    torch.cuda.synchronize(); t[i, 2] = time.perf_counter()

    if a.pipe == 'cpu':                     # CPU-only: all steps on the CPU, transfer at the end
        img = to_dtype(img)                                         # 3. Dtype
        torch.cuda.synchronize(); t[i, 3] = time.perf_counter()
        img = img.mul_(INV255)                                      # 4. Norm
        torch.cuda.synchronize(); t[i, 4] = time.perf_counter()
        img = img.permute(2, 0, 1).contiguous()                     # 5. CHW
        torch.cuda.synchronize(); t[i, 5] = time.perf_counter()
        img = img.unsqueeze(0)                                      # 6. Batch
        torch.cuda.synchronize(); t[i, 6] = time.perf_counter()
        img = img.to(device)                                        # 7. Trans
        torch.cuda.synchronize(); t[i, 7] = time.perf_counter()
    else:                                   # GPU-integrated: transfer first, then GPU steps
        img = img.to(device)                                        # 3. Trans
        torch.cuda.synchronize(); t[i, 3] = time.perf_counter()
        img = to_dtype(img)                                         # 4. Dtype
        torch.cuda.synchronize(); t[i, 4] = time.perf_counter()
        img = img.mul_(INV255)                                      # 5. Norm
        torch.cuda.synchronize(); t[i, 5] = time.perf_counter()
        img = img.permute(2, 0, 1)                                  # 6. CHW
        torch.cuda.synchronize(); t[i, 6] = time.perf_counter()
        img = img.unsqueeze(0)                                      # 7. Batch
        torch.cuda.synchronize(); t[i, 7] = time.perf_counter()

names = (['RGB', 'Tensor', 'Dtype', 'Norm', 'CHW', 'Batch', 'Trans'] if a.pipe == 'cpu'
         else ['RGB', 'Tensor', 'Trans', 'Dtype', 'Norm', 'CHW', 'Batch'])
d = np.diff(t[a.warmup:], axis=1) * 1000    # ms, steady-state runs only
med = np.median(d, axis=0)
print('#', '  '.join(f'{n:>7s}' for n in names), '    Total')
print('#', '  '.join(f'{m:7.3f}' for m in med), f'  {med.sum():7.3f}')
rgb = d[:, 0]
print(f'RESULT pipe={a.pipe:3s} cv={str(a.cv):4s} torch={str(a.torch):4s} '
      f'OMP_WAIT_POLICY={os.environ.get("OMP_WAIT_POLICY")} | '
      f'cvtColor P50={np.median(rgb):.3f} P95={np.percentile(rgb, 95):.3f} ms  (n={len(rgb)})')
