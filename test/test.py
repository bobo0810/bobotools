import os
import sys
import glob
import cv2

rootpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootpath + "/../")
sys.path.extend(glob.glob(rootpath + "/../*"))
import torch
from PIL import Image
import torchvision.models as models
from bobotools.torch_tools import Torch_Tools

"""
pytest自动化测试
"""
# =================torch工具类==============================
def test_get_model_info():
    """
    测试 获取模型信息
    """
    model = models.resnet18(pretrained=False)
    model_info = Torch_Tools.get_model_info([1, 3, 224, 224], model)
    print(model_info)


def test_vis_cam():
    """
    测试 可视化注意力图
    """
    from pytorch_grad_cam.utils.image import preprocess_image

    img_path = os.path.join(rootpath, "catdog.png")
    cv2_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(cv2_img)

    model = models.resnet50(pretrained=True)

    img = preprocess_image(img)
    img = torch.cat([img, img.clone()])  # [2,3,224,224]

    img_cam = Torch_Tools.vis_cam(model, img, pool_name="avgpool")

    cv2.imwrite(os.path.join(rootpath, "vis_cam.jpg"), img_cam)


def test_vis_tensor():
    """
    测试 可视化tensor
    """
    from pytorch_grad_cam.utils.image import preprocess_image

    img_path = os.path.join(rootpath, "catdog.png")
    cv2_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(cv2_img)

    img = preprocess_image(img)
    img = torch.cat([img, img.clone()])  # [2,3,224,224]
    vis_img = Torch_Tools.vis_tensor(img)

    cv2.imwrite(os.path.join(rootpath, "vis_img.jpg"), vis_img)
test_vis_tensor()