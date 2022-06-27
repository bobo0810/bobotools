from .com import get_model_size,get_model_time,get_model_complexity
import sys
class Torch_Tools(object):
    """
    Pytorch操作
    """

    def __init__(self):
        pass
    
    @staticmethod
    def get_model_info(input_shape, model):
        """
        获取模型信息，包括模型大小、前向推理耗时等
        
        input_shape: 输入形状 eg:[1,3,224,224]
        model: 模型
        """
        result_dict = {"input_shape": input_shape}

        # 获取模型大小
        size_dict=get_model_size(model)
        result_dict.update(size_dict)

        # 获取模型复杂度
        complex_dict=get_model_complexity(input_shape, model)
        result_dict.update(complex_dict)

        # 前向推理耗时
        time_dict=get_model_time(input_shape, model)
        result_dict.update(time_dict)
        
        return result_dict
    
    @staticmethod
    def tensor2img(
        tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), BCHW2BHWC=False
    ):
        """
        Tenso恢复为图像，用于可视化
        反归一化、RGB->BGR

        tensor: Tensor,形状[B,C,H,W]
        BCHW2BHWC: (可选)是否交换Tensor维度

        返回值
        imgs: Tensor 默认[B,C,H,W]。当BCHW2BHWC=Ture,则返回[B,H,W,C]
        """
        B, C, H, W = tensor.shape

        t_mean = torch.FloatTensor(mean).view(C, 1, 1).expand(3, H, W)
        t_std = torch.FloatTensor(std).view(C, 1, 1).expand(3, H, W)

        tensor = tensor * t_std.to(tensor) + t_mean.to(tensor)  # 反归一化
        tensor = tensor[:, [2, 1, 0], :, :]  # RGB->BGR
        if BCHW2BHWC:
            tensor = tensor.permute(0, 2, 3, 1)
        return tensor