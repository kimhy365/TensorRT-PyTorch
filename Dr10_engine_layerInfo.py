import torch
import logging
from torch2trt import torch2trt
from torch2trt import TRTModule
from si02_model_ver9 import ClassifierNet, LocalizerNet, ClassifierNetV2, LocalizerNetV2, \
LocalizerNet3Dsconv, OnlyClassifierNet, ClassifierNetVf
from myutils import load_model, convert_ndarray_to_tensor, load_label, track_time


logging.basicConfig(level=logging.DEBUG)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("\nLoading the ACNN model ...")
acnn_path = "/home/jetson/onnx2trt-env/2024_0926_0613_64_best_classifier.pth"     # ln_opt & cn
acnn = load_model(acnn_path, device='cpu', half=False)
# base_path = "/home/jetson/onnx2trt-env/2024_0913_2055_base_localizer.pth"        # ln_base
# base_ln = load_model(base_path, device=device, half=True)
# best_path = "/home/jetson/onnx2trt-env/2024_0912_1544_best_localizer.pth"         # ln_best
# best_ln = load_model(best_path, device=device, half=False)

img = torch.randn(16, 3, 480, 640, device='cpu')

try:
    # trt_model = torch2trt(acnn, [img], fp16_mode=False)
    onnx_model = torch.onnx.export(acnn, img, 'acnn_rev.onnx', opset_version=15)
except Exception as e:
    print(type(e).__name__)
    print(e)
        


""" 입력 이미지가 FP16인지를 조사
m = TRTModule()
# m.load_state_dict(torch.load('ln_trt_dict_fp16.pth'))
# m.load_state_dict(torch.load('ln_best_trt_fp16.pth'))
# m.load_state_dict(torch.load('ln_base_trt_fp16.pth'))
m.load_state_dict(torch.load('cn_trt_dict_fp16.pth'))

e = m.engine
for i in range(e.num_bindings):
    print(e.get_binding_name(i), e.get_binding_dtype(i), e.binding_is_input(i))
    
    
# 결과 ======================
# input_0 DataType.HALF True
# output_0 DataType.HALF False
"""
