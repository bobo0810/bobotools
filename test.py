import torchvision.models as models
from bobotools.torch_tools import Torch_Tools

#=================Torch工具类==============================
def test_get_model_info():
    model = models.resnet18(pretrained=False)
    model_info=Torch_Tools.get_model_info([1,3,224,224],model)
    print(model_info)