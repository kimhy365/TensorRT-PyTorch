import cv2
import numpy as np
import torch
from torch2trt import torch2trt
from torch2trt import TRTModule

from Dr09_model_info import ClassifierNet, LocalizerNet, ClassifierNetV2, LocalizerNetV2, \
LocalizerNet3Dsconv, OnlyClassifierNet, ClassifierNetVf, ClassifierNetVfOpt


USE_FP16 = True

# 모델 불러오기
acnn = ClassifierNetVfOpt().cuda().eval();     print('ACNN-opt-w/propre')

# 입력 데이터 생성
# dummy_input = torch.randn(1, 3, 480, 640).to(torch.uint8).cuda()

# 1. 가상의 이미지 생성 (실제 cv2.imread 결과와 동일한 (H, W, C), uint8 상태)
img_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# 2. 색상 공간 변환 (BGR -> RGB)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. Tensor 변환 + 차원 재배치 (HWC -> CHW) + Batch 차원 추가
dummy_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

# 4. GPU 전송 (이때 비동기 전송 옵션 테스트도 겸할 수 있습니다)
dummy_input = dummy_img.to('cuda', non_blocking=True)

# # --- 최종 검증 ---
# print(f"Shape: {dummy_input.shape}")  # 기대값: torch.Size([1, 3, 480, 640])
# print(f"Dtype: {dummy_input.dtype}")  # 기대값: torch.uint8


# torch2trt 변환 (fp16_mode 필수)
ln = acnn.localizer
ln_trt = torch2trt(ln, [dummy_input], fp16_mode=USE_FP16)    # log_level=trt.Logger.VERBOSE # 필요시 로그 확인

# 저장
ln_trt_path = 'classifier_with_prepro_fp16.pth'
torch.save(ln_trt.state_dict(), ln_trt_path)

# [TensorRT] WARNING: only a single logger pointer at any given time
ln_trt = TRTModule()
ln_trt.load_state_dict(torch.load(ln_trt_path))
ln_trt.eval()
cn_trt = TRTModule()
cn_trt.load_state_dict(torch.load(cn_trt_path))
cn_trt.eval()

# TensorRT engine 추출 및 저장
# print(dir(cn_trt))
engine_ln = ln_trt.engine
with open(ln_trt_path.replace('pth', 'engine'), 'wb') as f:
    f.write(engine_ln.serialize()) 
# engine_cn = cn_trt.engine
# with open(cn_trt_path.replace('pth', 'engine'), 'wb') as f:
    # f.write(engine_cn.serialize()) 
print("TensorRT engine saved ...")

# Profiling
# /usr/src/tensorrt/bin/trtexec --dumpProfile --warmUp=5000 --iterations=100 --avgRuns=100 --loadEngine=ln_opt_fp16.engine 
sys.exit()

# 추론 루프
# 1. 카메라에서 BGR 이미지 읽기 (numpy uint8)
# 2. HWC -> CHW 변환 (이건 CPU에서 수행)
img_chw = torch.from_numpy(frame.transpose(2, 0, 1)).cuda().unsqueeze(0)

# 3. 모델 실행 (내부에 Dtype 변환, Norm, STN 모두 포함)
output = model_trt(img_chw)



import torch
from torch2trt import torch2trt


# 앞서 정의하신 ClassifierNetVfOpt 클래스가 로드되어 있다고 가정합니다.

def convert_all_models():
    mods = ['base', 'opt', 'best']
    fps = [32, 16]

    # [중요] 엔진 입구를 uint8로 고정하기 위한 더미 데이터
    # 이미지 사이즈가 (480, 640)이 맞는지 확인해 주세요.
    dummy_input = torch.randint(0, 255, (1, 3, 480, 640), dtype=torch.uint8, device='cuda')

    for mod in mods:
        # 1. 원본 파이토치 가중치 경로 (학습된 모델 가중치가 있다면)
        # weight_path = f'classifier_{mod}_trained.pth'

        for fp in fps:
            engine_name = f'engine_e2e_{mod}_fp{fp}.pth'
            print(f"[{mod} - FP{fp}] 변환 시작...")

            # 2. 모델 초기화 및 GPU 적재
            model = ClassifierNetVfOpt(fp=fp, mod=mod).eval().cuda()

            # 3. 학습된 가중치 로드 (주석 해제 후 사용)
            # 주의: base_theta 등 새로 추가된 버퍼 때문에 strict=False로 불러와야 에러가 안 납니다!
            # model.load_state_dict(torch.load(weight_path), strict=False)

            # 4. torch2trt 엔진 변환
            fp16_mode = (fp == 16)
            try:
                model_trt = torch2trt(
                    model,
                    [dummy_input],
                    fp16_mode=fp16_mode,
                    # max_workspace_size=1<<26, # 64MB (메모리 부족 에러 시 주석 해제)
                )

                # 5. 변환된 TensorRT 엔진 모델 저장
                torch.save(model_trt.state_dict(), engine_name)
                print(f" => 성공! [{engine_name}] 저장 완료\n")

            except Exception as e:
                print(f" => 실패! 변환 중 에러 발생: {e}\n")


if __name__ == '__main__':
    # 변환 전, CUDA 메모리 초기화
    torch.cuda.empty_cache()
    convert_all_models()
