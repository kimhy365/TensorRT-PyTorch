# Hybrid TensorRT–PyTorch Inference Acceleration

**Hybrid TensorRT–PyTorch Inference Acceleration for Attention-Based CNNs with Dynamic ROI Sampling: A Case Study on Jetson Nano**

Hyun-Yong Kim (Chungbuk National University)

➡️ [Read the preprint on Preprints.org](https://doi.org/10.20944/preprints202611.0026.v1) (DOI: 10.20944/preprints202611.0026.v1)
➡️ Base model: [Attention-based CNN (A-CNN)](https://github.com/kimhy365/AttentionCNN), IEEE TIM 2025

## 📚 Overview

TensorRT speeds up inference through layer fusion, lower precision and kernel auto-tuning. However, a model with even one unsupported operator cannot be compiled into a single TensorRT engine.

Our [A-CNN](https://github.com/kimhy365/AttentionCNN) extracts a region of interest (ROI) with `affine_grid()` and `grid_sample()`. These operators are not supported by the toolchain available on the Jetson Nano (JetPack 4.6, TensorRT 8.0.1, torch2trt).

This repository provides a **hybrid TensorRT–PyTorch deployment**:

- The **localization network (LN)** and **classification network (CN)** are compiled into TensorRT engines with torch2trt.
- The **ROI sampler** (only about 0.45 MFLOPs) falls back to native PyTorch.
- You don't need to write custom CUDA kernels or TensorRT plugins.

Key contributions:

- A hybrid TensorRT–PyTorch pipeline for A-CNN variants on the legacy Jetson Nano, using the graph-partitioning and PyTorch-fallback idea.
- A latency analysis of the full pipeline across preprocessing, device (CPU/GPU), framework (PyTorch/TensorRT) and precision (FP32/FP16).
- Evidence that lower FLOPs do not always mean lower latency: depthwise separable convolution (DSConv) gains little or nothing from TensorRT FP16 on the Jetson Nano.

## 🔧 Hybrid Inference Pipeline

```
frame ─► Preprocess (GPU) ─► LN [TensorRT] ─► ROI Sampler [PyTorch] ─► CN [TensorRT] ─► argmax
```

- **Preprocess:** `cv2.cvtColor` runs on the CPU. The raw `uint8` image is then sent to the GPU, where datatype conversion, normalization and layout changes are done.
- **LN variants:** `LN-base` (2.2 GFLOPs), `LN-k3` (0.98 GFLOPs, 3×3 kernels), `LN-dsc` (0.34 GFLOPs, DSConv).
- **Sampler:** `affine_grid()` + `grid_sample()` produce a 96×96 ROI patch.

## 🖥️ Environment

| Item | Version |
|---|---|
| Device | NVIDIA Jetson Nano 4GB (10W MAXN, `jetson_clocks`) |
| OS / Python | Ubuntu 20.04 / Python 3.8 (Q-engineering image, https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image) |
| PyTorch | 1.13 |
| CUDA / cuDNN | 10.2 / 8.2 |
| TensorRT | 8.0.1.6 |
| torch2trt | 0.5.0 (only `fp16_mode` set; all other arguments at their defaults) |
| OpenCV | 4.10 |
| Other packages | numpy, thop, torchinfo, torchsummary |

## 📊 Results

End-to-end latency of the ACNN variants on the Jetson Nano (Table 6 in the paper). Each value comes from 1,000 steady-state runs and includes preprocessing and postprocessing. E2E latency is measured from the return of `cv2.imread()` to the return of the classification result; image acquisition and disk I/O are excluded.

<p align="center"><img src="images/table6_e2e_latency.png" alt="Table 6: End-to-end latency of the ACNN variants" width="100%"></p>

**Notes:**

- **Hybrid<sup>a/b</sup>** compares PyTorch and hybrid runtimes at the same FP32 precision.
- **FP16<sup>b/c</sup>** is the extra gain from lowering precision inside the hybrid runtime.
- **Total<sup>a/c</sup>** is the overall speedup from PyTorch FP32 to hybrid FP16.

**Highlights:**

- The fastest configuration is **hybrid FP16 ACNN-k3 at 17.78 ms (56.2 fps)**, a 2.23× speedup over its PyTorch FP32 counterpart (39.65 ms).
- The deployed ACNN-dsc runs at 20.40 ms with hybrid FP32, a 1.88× speedup over its PyTorch FP32 counterpart (38.37 ms).
- GPU-integrated preprocessing reduced preprocessing time from 5.125 ms (CPU-only, FP32) to 2.116 ms (GPU, FP16), a 2.42× speedup. It also reduced latency jitter.
- Under TensorRT FP16, ACNN-k3 ran faster than ACNN-dsc even though ACNN-k3 has about 3× more FLOPs; the verbose build log shows that TensorRT kept the DSConv blocks of LN-dsc in FP32.
- **Accuracy:** the hybrid FP32 runtime gave the same predictions as PyTorch FP32 for the deployed ACNN-dsc, and FP16 changed the number of misclassified test images (out of 1,200) by at most two in every model variant.

> See the paper for stepwise profiling and the TensorRT engine and layer profiles.

## 📂 Repository Structure

The scripts are provided **as used in the experiments, without refactoring**. File paths are hard-coded to the original working directory on the Jetson Nano (`/home/jetson/onnx2trt-env/`); edit them, or place the files there, before running.

```
*.py, *.sh                 experiment scripts (flat, so that the original imports work)
checkpoints/               trained PyTorch models (whole-model pickles, loaded with torch.load)
  2024_0926_0613_64_best_classifier.pth   ACNN-dsc (LN-dsc + CN), deployed model
  2024_0913_2055_base_localizer.pth       LN-base
  2024_0912_1544_best_localizer.pth       LN-k3
  acnn_dsc_8runs/                         eight ACNN-dsc checkpoints used for the FP32/FP16 accuracy comparison
engines/                   TensorRT engines used in the measurements (Jetson Nano, TensorRT 8.0.1 only)
  *.engine                                serialized engines profiled with trtexec (Tables 7 and 8)
  torch2trt/*.pth                         torch2trt TRTModule state dicts used by the hybrid scripts
logs/
  trtexec/                                trtexec logs for Tables 7 and 8
  build_verbose/                          TensorRT verbose build logs (per-layer datatypes, Section 5.1)
  results/                                raw per-run outputs of the latency scripts
  accuracy/                               per-image accuracy outputs
images/                    figures for this README
```

The image dataset is not included.

The checkpoints are whole-model pickles; the scripts import the model classes from `si02_model_ver9.py` or `Dr09_model_info.py` before calling `torch.load`, so these files must be importable from the working directory. In the scripts, the model variants are named `base` = ACNN-base, `best` / `bst` = ACNN-k3, and `opt` / `dsc` = ACNN-dsc.

### Scripts and the results they produce

| Paper item | Script(s) | Raw output |
|---|---|---|
| Model definitions, parameters and FLOPs (Tables 1, 2) | `si02_model_ver9.py`, `Dr09_model_info.py`, `acnn_info.py`, `myutils.py` | – |
| TensorRT conversion (Section 3.2) | `convert2trt_0301.py` | `engines/torch2trt/` |
| Serialized engines for trtexec | `Dr08_Optim_infer.py` (first part), `Dr10_engine_layerInfo.py` | `engines/*.engine` |
| Preprocessing, stepwise (Table 3) | `Dr01_prepCPU.py` (CPU-only FP32), `Dr02_prepGPU2CPU.py` (CPU-only FP16), `Dr02_prepGPU.py` (GPU-integrated), `Dr00_repeat.py` (repeated runs) | `logs/results/profile-prepro-*.txt`, `exp_*.txt` |
| Preprocessing, E2E (Table 4, Figure 2) | `Dr02_prepGPU-1.py` | `logs/results/profile-prepro-*e2e*.txt` |
| Stepwise forwarding (Table 5) | `Dr06_infer_pt(cpu)_profile.py` (PyTorch CPU), `Dr06_infer_pt_profile.py` (PyTorch GPU), `Dr04_infer_rt_profile.py` (hybrid) | `logs/results/profile-pt(cpu)*.txt`, `profile-pt.txt`, `profile-{base,bst,opt}-fp{16,32}.txt` |
| E2E latency (Table 6, Figure 3) | `Dr07_infer_pt_e2e.py` (PyTorch), `Dr05_infer_rt_e2e.py` (hybrid) | `logs/results/profile-pt-e2e.txt`, `profile-trt-e2e.txt` |
| Classification accuracy (Section 4.2.2) | `Dr07_infer_pt_acc.py` (PyTorch), `phd8_trt_acc.py` (hybrid) | `logs/accuracy/` |
| Engine timing and layer profile (Tables 7, 8) | trtexec, see below | `logs/trtexec/` |
| Per-layer datatypes of the FP16 LN-dsc engine (Section 5.1) | trtexec / torch2trt verbose build | `logs/build_verbose/` |
| cv2.cvtColor() thread test | `thread_test.py`, `run_thread_test.sh` | – |

### trtexec commands (Tables 7 and 8)

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=<engine> --warmUp=5000 --iterations=100 --avgRuns=100 \
    --dumpProfile --separateProfileRun
```

`--separateProfileRun` keeps the layer profiler out of the latency measurement. Table 7 reports the medians; Table 8 reports the per-layer means printed by trtexec.

### Notes

- TensorRT engines are specific to the device and software stack on which they are built. The files in `engines/` run only on a Jetson Nano with TensorRT 8.0.1; on other systems, rebuild them with `convert2trt_0301.py`.
- The TensorRT builder selects the fastest tactic for each layer at build time, so engines built from the same model and settings can differ in their per-layer precision (see `logs/build_verbose/`).

## 📄 Citation

```bibtex
@article{kim2026hybrid,
title = {Hybrid TensorRT--PyTorch Inference Acceleration for Attention-Based CNNs with Dynamic ROI Sampling: A Case Study on Jetson Nano},
author = {Kim, Hyun-Yong},
journal = {Preprints},
year = {2026},
doi = {10.20944/preprints202611.0026.v1}
}
```

Base model:

Hyun-Yong Kim, Taek-Joon Yi, and Jong-Yun Lee, "An Attention-based Convolutional Neural Network with Spatial Transformer Module for Automated Optical Inspection of Small Objects," *IEEE Transactions on Instrumentation and Measurement*, 2025. DOI: [10.1109/TIM.2025.3548240](https://doi.org/10.1109/TIM.2025.3548240)
