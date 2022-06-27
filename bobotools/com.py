import torch
import time
import os
from ptflops import get_model_complexity_info
cur_path = os.path.abspath(os.path.dirname(__file__))


def get_model_size(model):
    '''
    获取模型大小（MB）
    '''
    model_path = os.path.join(cur_path, str(time.time())+"_temp.pt")
    torch.save(model, model_path)
    model_size = os.path.getsize(model_path) / float(1024 * 1024)
    os.remove(model_path)
    return {"model_size":str(round(model_size, 2))+" MB"}

def get_model_complexity(input_shape,model):
    '''
    获取模型复杂度，即参数量Params、计算量FLOPs
    '''
    B,C,H,W=input_shape
    flops, params = get_model_complexity_info(model, (C,H,W), as_strings=True,print_per_layer_stat=False, verbose=False)
    return {"FLOPs":flops,"Params":params}         

@torch.no_grad()
def get_model_time(input_shape,model,warmup_nums=100, iter_nums=300):
    '''
    获取模型前向耗时
    '''
    time_dict={}

    img = torch.ones(input_shape)

    # 记录原模型状态
    ori_flag = model.training  
    ori_device= next(model.parameters()).device

    model.eval()

    device_list = ["cuda:0","cpu"] if torch.cuda.is_available() else ["cpu"]
    for device in device_list:
        img = img.to(device)
        model.to(device)

        # 预热
        for _ in range(warmup_nums):
            model(img)
            if "cuda" in device:
                torch.cuda.synchronize()
        # 正式
        start = time.time()
        for _ in range(iter_nums):
            model(img)
            # 每次推理，均同步一次。算均值
            if "cuda" in device:
                torch.cuda.synchronize()
        end = time.time()
        total_time = ((end - start) * 1000) / float(iter_nums)
        time_dict[device]=str(round(total_time, 2))+" ms"
    # 恢复到原状态
    model.training = ori_flag  
    model.to(ori_device)
    return time_dict
