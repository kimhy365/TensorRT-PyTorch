# Hybrid TensorRT–PyTorch Inference Acceleration

Code, checkpoints, and raw logs for the paper

> H.-Y. Kim, "Hybrid TensorRT–PyTorch Inference Acceleration for Attention-Based CNNs with Dynamic ROI Sampling: A Case Study on Jetson Nano." Preprint: https://doi.org/10.20944/preprints202611.0026.v1 (revised version under review at *Sensors*, manuscript ID sensors-4695531)

The base model is the attention-based CNN (ACNN) of Kim, Yi, and Lee, *IEEE Transactions on Instrumentation and Measurement*, 2025, https://doi.org/10.1109/TIM.2025.3548240.

## Overview

TensorRT accelerates inference, but a model containing even one unsupported operator cannot be compiled into a single engine. The ROI sampler of the ACNN uses `affine_grid()` and `grid_sample()`, which are not supported by the Jetson Nano toolchain (JetPack 4.6, TensorRT 8.0.1, torch2trt). The model is therefore executed in a hybrid manner:

- the localization network (LN) and the classification network (CN) run as TensorRT engines built with torch2trt;
- the ROI sampler falls back to native PyTorch.

```
Frame → Preprocessing (GPU) → LN [TensorRT] → ROI sampler [PyTorch] → CN [TensorRT] → argmax
```

LN variants: LN-base, LN-k3, and LN-dsc (depthwise separable convolutions). The sampler produces a 96×96 ROI patch.

## Main results (Jetson Nano, 10 W MAXN)

- Hybrid FP16 ACNN-k3 (fastest configuration): 17.78 ms (56.2 fps), a 2.23× speedup over its PyTorch FP32 counterpart (39.65 ms).
- Hybrid FP32 ACNN-dsc (deployed model): 20.40 ms, a 1.88× speedup over its PyTorch FP32 counterpart (38.37 ms).
- Moving most preprocessing steps to the GPU reduced the preprocessing latency from 5.125 ms to 2.116 ms.
- FP16 changed the number of misclassified test images (out of 1,200) by at most two in every model variant.
- Under FP16, ACNN-k3 ran faster than ACNN-dsc despite about three times higher FLOPs; the verbose build log shows that TensorRT kept the DSConv blocks of LN-dsc in FP32.

![Table 6](images/table6_e2e_latency.png)
<!-- TODO(author): check that this image matches the revised Table 6 -->

## Environment

| Item | Version / setting |
|---|---|
| Device | NVIDIA Jetson Nano 4 GB, 10 W (MAXN) mode, clocks fixed with `sudo jetson_clocks` |
| OS | Ubuntu 20.04 image for Jetson Nano by Q-engineering (https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image) |
| Python | 3.8 |
| PyTorch | 1.13 |
| CUDA / cuDNN | 10.2 / 8.2 |
| TensorRT | 8.0.1.6 |
| torch2trt | 0.5.0 (only `fp16_mode` set; all other arguments at their defaults) |
| OpenCV | 4.10 |
| Other packages | numpy, thop, torchinfo, torchsummary |

## Repository layout

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

The image dataset is not included. <!-- TODO(author): state the reason / availability of the dataset -->

The checkpoints are whole-model pickles; the scripts import the model classes from `si02_model_ver9.py` or `Dr09_model_info.py` before calling `torch.load`, so these files must be importable from the working directory.

## Scripts and the results they produce

<!-- TODO(author): confirm the script-to-table mapping below -->

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

In the scripts, the model variants are named as follows: `base` = ACNN-base, `best` / `bst` = ACNN-k3, `opt` / `dsc` = ACNN-dsc.

## trtexec commands (Tables 7 and 8)

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=<engine> --warmUp=5000 --iterations=100 --avgRuns=100 \
    --dumpProfile --separateProfileRun
```

`--separateProfileRun` keeps the layer profiler out of the latency measurement. Table 7 reports the medians; Table 8 reports the per-layer means printed by trtexec.

## Notes

- TensorRT engines are specific to the device and software stack on which they are built. The files in `engines/` run only on a Jetson Nano with TensorRT 8.0.1; on other systems, rebuild them with `convert2trt_0301.py`.
- The TensorRT builder selects the fastest tactic for each layer at build time, so engines built from the same model and settings can differ in their per-layer precision (see `logs/build_verbose/`).
- E2E latency is measured from the return of `cv2.imread()` to the return of the classification result; image acquisition and disk I/O are excluded.

## Citation

<!-- TODO(author): paste the BibTeX entries (this paper and the base ACNN model) from the previous README -->

## License

<!-- TODO(author): choose a license (e.g., MIT) -->
