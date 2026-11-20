import os

for i in range(10):
    print()
    # print(i+1, 'Pytorch')
    # os.system('python paper_infer_acnn_v5.py')
    
    # print(i+1, 'TensorRT')
    # os.system('python phd6_infer_acnn_trt.py')
    
    # print(i+1, 'Preprocessing on CPU with permute.contiguous()')
    # os.system("python Dr01_prepCPU.py | tee -a results/exp_$(date +'%Y%m%d')_prepCPU.txt")
    
    # print(f'#{i+1} Preprocessing on CPU chaining with permute()')
    # os.system("python Dr01_prepCPU.py | tee -a results/exp_$(date +'%Y%m%d')_prepCPU_chain2.txt")
    
    print(f'#{i+1} Preprocessing on CPU')
    os.system("python Dr02_prepGPU2CPU.py")
    
    # print(f'#{i+1} Preprocessing on GPU')
    # os.system("python Dr02_prepGPU.py")
