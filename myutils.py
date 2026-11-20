# =========================================================================
# Deep learning 유틸리티
# 작성일 : 2023-09-30
# =========================================================================
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary
from torchinfo import summary as summary2
import tensorrt as trt
import time


# TensorRT 추론에 필요한 함수들 --------------------------------------------------------------------------------------

# TensorRT 엔진 로드 함수
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # 단일 전역 TensorRT Logger 설정 (WARNING)
def load_engine(engine_path):  
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

#이미지를 불러와서 TensorRT 모델에 맞게 전처리
def preprocess_image(image_path, dtype=np.float32):
    img = cv2.imread(image_path)  # BGR 형식으로 로드
    t0 = time.time()
    img = img[:, :, ::-1].transpose((2, 0, 1)).astype(dtype) / 255.0
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB 변환
    # img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)
    # img = img.astype(dtype) / 255.0  # 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가 (1, C, H, W)
    # img = np.ascontiguousarray(img)
    print('이미지 변환시간:', (time.time() - t0)*1000, 'ms')
    return img
    
    	# # preprocessing
	
	# img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
	# img = img.to(device)
    

def track_time(start_time, stage_name):
    elapsed_time = time.time() - start_time
    print(f" - {elapsed_time * 1000:5.2f} ms : {stage_name}")
    return time.time()  # 반환값은 새로운 시작 시간


# YOLO 주석파일 불러오기
def load_label(lbl_path, resolution=None):
    with open(lbl_path, 'r') as f:
        data = f.read().split()
        if resolution is None:
            cx, cy = float(data[1]), float(data[2])
        else:
            cx, cy = round(float(data[1]) * resolution[0]), round(float(data[2]) * resolution[1])
        return int(data[0]), cx, cy 


# Convert tensor image to ndarray image
def convert_tensor_to_ndarray(img_t):
    """pytorch의 tensor 이미지를 ndarray 형식으로 변환
    :param img_t: [1, ch, nr, nc]
    :return:
    """
    img = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 배치 차원 제거 후 [H, W, C]로 변환
    img = (img * 255).astype('uint8')  # [0, 1] 범위 값을 [0, 255] 범위로 변환
    # img = img_t.squeeze().numpy().transpose(1, 2, 0) * 255     # del bs -> [h, w, ch] -> np ->
    # img = img[:, :, ::-1].astype('uint8')
    return img


def convert_tensor_image_to_opencv(img, bgr2rgb=True):
    """Convert a Tensor to numpy image.
    numpy 변환 -> [0, 255] -> 'uint8' -> (H, W, C) -> BGR2RGB
    """
    img = (img.cpu().numpy() * 255).astype('uint8').transpose((1, 2, 0))
    if bgr2rgb:
        img = img[:, :, ::-1]

    # 연속된 메모리로 변환
    """ cv2.rectangle 함수에서 Error 발생 
    - Layout of the output array img is incompatible with cv::Mat
    - PyTorch의 Tensor는 다양한 연산을 거친 후 메모리에서 비연속적인(strided) 형태가 될 수 있어. 
      이 상태에서 .numpy()를 호출하면 내부적으로 NumPy 배열로 변환되지만, 
      메모리 정렬 방식이 PyTorch와 다를 수 있어서 OpenCV 등에서 예상치 못한 오류가 발생할 수 있어."""
    img = np.ascontiguousarray(img)


    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return img


def convert_ndarray_to_tensor(img_np, bgr2rgb=True, gpu=False):
    if bgr2rgb:
        img_np = img_np[:, :, ::-1]                 # BGR2RGB
    img_np = img_np.transpose((2, 0, 1)).copy()            # [c, w, h]
    img_t = torch.from_numpy(img_np).float().div(255.0).unsqueeze(0)  # ToTensor: [0, 1] + batch

    if gpu:
        if torch.cuda.is_available():
            img_t = img_t.cuda()  # GPU 이동
        else:
            raise RuntimeError("GPU를 사용할 수 없습니다. CUDA가 활성화되어 있지 않습니다.")
    return img_t


def model_info(net, img_w=224, img_h=224):
    print(net)
    summary(net, input_size=(3, img_h, img_w), device='cpu')
    summary2(net)

    # 연산량 계산
    dummy_img = torch.randn(1, 3, img_h, img_w)
    flops, params = profile(net, inputs=(dummy_img,))
    print('Network : {:,.0f} Params, {:.2f} MFLOPs'.format(params, flops / 1e6))


def load_model(model_path, device='cpu', half=False):
    model = torch.load(model_path, map_location='cpu')
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Gets rid of the DataParallel wrapper

    """
    # # weight 파일인 경우 : 모델 구조(config) + 가중치(weights)
    # weight_path = 'best_weights.pth'
    # 
    # 1단계 : 모델 구조 생성
    # model = torchvision.models.efficientnet_b0()
    # num_in_features = model.classifier[1].in_features
    # prob = model.classifier[0].p
    # model.classifier = nn.Sequential(
    #     # nn.BatchNorm1d(num_features=num_in_features, eps=1e-05, momentum=0.1),
    #     nn.Dropout(p=prob, inplace=True),
    #     nn.Linear(in_features=num_in_features, out_features=2, bias=True),
    #     nn.Softmax(dim=1))
    # summary(model, (3, 224, 224), device='cpu')     # device='gpu'에서는 오류 발생
    # 
    # # 2단계 : 가중치 불러오기
    # model.load_state_dict(torch.load(weight_path))  # size mismatch
    """

    # 추론을 위한 모델 준비작업 : eval mode + gradient 비활성화
    model.eval()
    for param in model.parameters():  # inference time : 5ms  vs 7.5ms
        param.requires_grad = False  # Sets the parameters to not required gradients

    if half:
        return model.half().to(device)
    else:
        return model.to(device)


# Loss, Accuracy 2개의 그래프 작성
def display_training_graph_v2(train_loss, train_acc, test_loss, test_acc, savefile=None):
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

    ax[0].plot(epochs, train_loss, 'b.-', label='Train Loss')
    ax[0].plot(epochs, test_loss, 'r.:', label='test Loss')
    ax[0].set_xlim(0, len(train_loss))
    ax[0].set_yscale('log')
    # ax[0].xaxis.set_major_locator(mticker.MultipleLocator(5))  # 5의 배수
    # ax[0].xaxis.set_minor_locator(mticker.MultipleLocator(1))
    # ax[0].yaxis.set_major_locator(mticker.MaxNLocator(5))
    # ax[0].yaxis.set_minor_locator(mticker.MaxNLocator(5))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Loss')
    ax[0].minorticks_on()
    ylim = ax[0].get_ylim()
    ax[0].set_ylim(np.max([0, ylim[0]]), ylim[1])  # 0보다 작은 눈금이 나오지 않게 함
    # ax[0].set_xticks(plt.gca().get_yticks(), minor=True)
    # ax[0].set_yticks(plt.gca().get_yticks(), minor=True)
    ax[0].grid(True, linewidth=0.5, alpha=0.5)    # color='black', linestyle='--', which='both' (주축과 보조축)

    ax[1].plot(epochs, train_acc, 'b.-', label='Train accuracy')
    ax[1].plot(epochs, test_acc, 'r.:', label='Test accuracy')
    ax[1].set_xlim(0, len(train_acc))
    # ax[1].xaxis.set_major_locator(mticker.MultipleLocator(5))  # 5의 배수
    # ax[1].xaxis.set_minor_locator(mticker.MultipleLocator(1))
    # ax[1].yaxis.set_major_locator(mticker.MaxNLocator(5))
    # ax[0].yaxis.set_minor_locator(mticker.MaxNLocator(5))
    # ax[1].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    # ax[1].yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='lower right')
    ax[1].set_title('Accuracy')
    ax[1].minorticks_on()
    ylim = ax[1].get_ylim()
    ax[1].set_ylim(np.max([0, ylim[0]]), np.min([ylim[1], 1]))  # 100%가 넘는 눈금이 나오지 않게 함
    ax[1].set_yticks(plt.gca().get_yticks())
    # print('plt.gca():', plt.gca().get_yticks().tolist())  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # axes.set_yticks(axes.get_yticks().tolist())
    ax[1].set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    ax[1].grid(True, linewidth=0.5, alpha=0.5)
    """
    axes.title.set_size(20)
    axes.xaxis.label.set_size(16)
    axes.yaxis.label.set_size(16)
    """

    if savefile is None:
        plt.show(block=False)
        plt.pause(1)
    else:
        plt.savefig(savefile)
        print("Saved graph to '{}'".format(savefile))


# Loss의 그래프 작성
def display_training_graph(train_loss, test_loss, savefile=None):
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)

    ax.plot(epochs, train_loss, 'b.-', label='Train Loss')
    ax.plot(epochs, test_loss, 'r.:', label='Test Loss')
    ax.set_xlim(0, len(train_loss))
    ax.set_yscale('log')
    # ax[0].xaxis.set_major_locator(mticker.MultipleLocator(5))  # 5의 배수
    # ax[0].xaxis.set_minor_locator(mticker.MultipleLocator(1))
    # ax[0].yaxis.set_major_locator(mticker.MaxNLocator(5))
    # ax[0].yaxis.set_minor_locator(mticker.MaxNLocator(5))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.set_title('Training, Validation and Test Losses')
    ax.minorticks_on()
    ylim = ax.get_ylim()
    ax.set_ylim(np.max([0, ylim[0]]), ylim[1])  # 0보다 작은 눈금이 나오지 않게 함
    # ax[0].set_xticks(plt.gca().get_yticks(), minor=True)
    # ax[0].set_yticks(plt.gca().get_yticks(), minor=True)
    ax.grid(True, linewidth=0.5, alpha=0.5)    # color='black', linestyle='--', which='both' (주축과 보조축)

    if savefile is None:
        plt.show(block=False)
        plt.pause(1)
    else:
        plt.savefig(savefile)
        print("Saved graph to '{}'".format(savefile))
