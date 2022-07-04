import hashlib
from tqdm import tqdm
import numpy as np
import cv2
import torch
import base64
import urllib.request
import os

class Img_Tools(object):
    """
    Img工具类
    """

    def __init__(self):
        pass
    
    @staticmethod
    def read_web_img(image_url=None,image_file=None,image_base64=None,url_time_out=10):
        '''
        参数三选一，当传入多个参数，仅返回最高优先级的图像

        优先级： 文件 > base64 > url 
        
        url_time_out : URL下载耗时限制，默认10秒
        '''
        if image_file:
            try:
                img = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)
                if img.any():
                    return img
                else:
                    return 'IMAGE_ERROR_UNSUPPORTED_FORMAT'
            except:
                return 'IMAGE_ERROR_UNSUPPORTED_FORMAT'

        elif image_base64:
            try:
                img = base64.b64decode(image_base64)
                img_array = np.frombuffer(img, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img.any():
                    return img
                else:
                    return 'IMAGE_ERROR_UNSUPPORTED_FORMAT'
            except:
                return 'IMAGE_ERROR_UNSUPPORTED_FORMAT'
        elif image_url:
            try:
                resp = urllib.request.urlopen(image_url,time_out=url_time_out)
            except:
                return 'URL_DOWNLOAD_TIMEOUT'
            try:
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                if img.any():
                    return img
                else:
                    return 'IMAGE_ERROR_UNSUPPORTED_FORMAT'
            except:
                return 'INVALID_IMAGE_URL'
        else:
            return 'MISSING_ARGUMENTS'

    @staticmethod
    def filter_md5(imgs_list, compare_list=[]):
        """
        基于md5对imgs_list图像自身去重，返回重复图像的列表
        compare_list(可选): 基于对比库再去重。

        若文件读取出错，则过滤掉
        """
        ###########################内部方法######################################
        def get_md5(file):
            """计算md5"""
            file = open(file, "rb")
            md5 = hashlib.md5(file.read())
            file.close()
            return md5.hexdigest()

        def get_md5lib(imgs_list):
            """构建md5底库"""
            print("start generate md5lib...")
            md5_lib = []
            for img_path in tqdm(imgs_list):
                try:
                    md5_lib.append(get_md5(img_path))
                except:
                    continue
            return md5_lib

        def query_md5(imgs_list, md5_lib):
            print("start filter md5...")
            rm_list = []
            for img_path in tqdm(imgs_list):
                try:
                    md5 = get_md5(img_path)
                    if md5 not in md5_lib:
                        md5_lib.append(md5)
                    else:
                        rm_list.append(img_path)
                except:
                    continue
            return rm_list

        #################################################################

        assert type(imgs_list) is list
        assert type(compare_list) is list
        md5_lib = []
        # 构建对比库
        if len(compare_list) > 0:
            md5_lib.extend(get_md5lib(compare_list))
        return query_md5(imgs_list, md5_lib)

    @staticmethod
    def verify_integrity(imgs_list):
        """
        验证图像完整性

        imgs_list(list): 包含图像绝对路径的列表。 eg:[/home/a.jpg,...]

        return
        error_list(list): 错误图像的列表。eg:[/home/a.jpg,...]
        """
        error_list = []
        for img_path in tqdm(imgs_list):
            try:
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                assert type(image) is np.ndarray
            except:
                error_list.append(img_path)
        return error_list

    @staticmethod
    def plot_yolo(img_path,txt_path,class_list,save_path,vis_conf=0.,lw=6):
        '''
        可视化yolo结果
        :param img_path: 图像路径 eg:/home/xxx.jpg
        :param txt_path: 文件路径  eg:/home/xxx.txt
        :param class_list: 类别列表 eg:['cat','dog']
        :param save_path: 保存路径  eg:/home/
        :param vis_conf: 超过该置信度阈值再可视化.(txt存在conf时生效)
        :param lw: 文字大小
        '''
        assert os.path.exists(img_path)
        assert os.path.exists(txt_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 读取图像
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = np.ascontiguousarray(img)  # 内存连续
        color, txt_color = (0, 0, 255), (0, 0, 255)  # 锚框、文字颜色
        tl = round(0.002 * (width + height) / 2) + 1  # 锚框、文字的粗细

        with open(txt_path, "r") as f:
            annotations = f.readlines()
            for ann in annotations:
                ann = list(map(float, ann.split()))
                # 过滤标签内空格行
                if len(ann) == 0:
                    continue
                ann[0] = int(ann[0]) 
                if len(ann) == 6:
                    cls, x, y, w, h, conf = ann
                    class_info=class_list[cls] + "_" + str(round(conf, 2)) 
                    if conf < vis_conf:
                        continue
                elif len(ann) == 5: 
                    cls, x, y, w, h = ann
                    class_info=class_list[cls] 
               
                c1 = int((x - w / 2) * width), int((y - h / 2) * height)
                c2 = int((x + w / 2) * width), int((y + h / 2) * height)
                # 绘制锚框
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  
                # 绘制文本
                cv2.putText(
                    img,
                    class_info,
                    (c1[0], c1[1] - 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tl,
                    lineType=cv2.LINE_AA,
                )  
        img_name=os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_path, img_name), img)
