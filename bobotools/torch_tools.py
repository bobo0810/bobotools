import sys
import numpy as np
from .com import get_model_size,get_model_time,get_model_complexity
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image
import torch
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
    
    @staticmethod
    def vis_cam(model, img_tensor, pool_name="global_pool"):
        """
        可视化注意力图
        建议: 请打印网络结构,确认当前模型的全局池化层名称, 赋值pool_name即可。

        img_tensor(tensor): 网络的输入Tensor  shape[B,C,H,W]
        pool_name(str): 可视化特征图的网络位置的名称。
            通常选取卷积网络最后输出的特征图  (卷积网络->全局池化->分类网络)
            默认timm库的全局池化名称为"global_pool",自定义模型需自行确定

        返回img_cam(numpy) [H,W,C] cv2.imwrite直接保存即可。
        更多可视化算法,请访问 https://github.com/jacobgil/pytorch-grad-cam
        """
        # 定位可视化层
        modules_list = []
        for name, module in model.named_modules():
            if pool_name in name:  
                break
            modules_list.append(module)
        target_layers = [modules_list[-1]]  # 全局池化层的前一层

        # 获取类激活映射
        with GradCAM(model,target_layers,use_cuda = True if torch.cuda.is_available() else False) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(
                input_tensor=img_tensor,
                targets=None,       # 默认基于模型预测最高分值的类别可视化
                aug_smooth=True,    # 平滑策略1
                eigen_smooth=True,  # 平滑策略2
            )

        # 拼成网格
        result=[]
        for i in range(len(img_tensor)):
            cam_i=grayscale_cam[i] # [224,224]
            img_i=img_tensor[i].numpy() # [3,224,224] [C,H,W] BGR通道

            img_i = deprocess_image(img_i) # (归一化、减均值、除方差)的逆操作
            img_i = np.transpose(img_i,(1,2,0)) /255  # 转为[H,W,C]
            img_cam = show_cam_on_image(img_i, cam_i, use_rgb=False) # 映射到原图
            result.append(img_cam) 
        return np.concatenate(result,axis=1)  # 合并为网格图