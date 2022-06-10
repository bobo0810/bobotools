import torch
import time
class Torch_Tools(object):
    """
    Pytorch操作
    """

    def __init__(self):
        pass
    
    @staticmethod
    def cal_model_time(input_shape, model, warmup_nums=100, iter_nums=300):
        """
        统计 模型前向耗时
        
        input_shape: 输入形状 eg:[1,3,224,224]
        model: 模型
        warmup_nums: 预热次数
        iter_nums: 总迭代次数，计算平均耗时
        """
        img = torch.ones(input_shape)

        device_list = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
        time_dict = {"input_shape": input_shape}
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
            for _ in tqdm(range(iter_nums)):
                model(img)
                # 每次推理，均同步一次。算均值
                if "cuda" in device:
                    torch.cuda.synchronize()
            end = time.time()
            total_time = ((end - start) * 1000) / float(iter_nums)

            time_dict[device] = "%.2f ms/img" % total_time
        return time_dict