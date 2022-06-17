from .com import get_model_size,get_model_time
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
        
        # 前向推理耗时
        time_dict=get_model_time(input_shape, model)
        result_dict.update(time_dict)
        
        return result_dict