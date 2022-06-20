import os
current_dir = os.path.abspath(os.path.dirname(__file__))
os.system(' export PYTHONPATH="%s:$PYTHONPATH"'%current_dir)

import torchvision.models as models
from .torch_tools import Torch_Tools
from .img_tools import Img_Tools
'''
pytest自动化测试
'''
#=================torch工具类==============================
def test_get_model_info():
    model = models.resnet18(pretrained=False)
    model_info=Torch_Tools.get_model_info([1,3,224,224],model)
    print(model_info)

#=================img工具类==============================
def test_plot_yolo():
    img_path = current_dir + '../test/sample.jpeg'
    txt_path =  current_dir + '../test/sample.txt'
    class_list = ['box']
    save_path = current_dir
    Img_Tools.plot_yolo(img_path,txt_path,class_list,save_path)