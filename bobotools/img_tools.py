import hashlib
from tqdm import tqdm
import numpy as np
import cv2
import torch
import base64
from multiprocessing import Process
from urllib.request import urlretrieve
import os
from .list_tools import List_Tools


class Img_Tools(object):
    """
    Img工具类
    """

    def __init__(self):
        pass
    @staticmethod
    def download_url(url_path_list,process_nums=10):
        """
        多进程下载图片URL
        url_path_list(list): url和图像保存路径对应的列表
            eg: [
                    {"url":"https://xxxx","path":"/home/xxx/abc.jpg"},
                    ...
                ]
        process_nums(int): 进程数
        """
        # ----------------------------内部方法-------------------------------------
        class DownloadProcess(Process):  # 继承Process类
            def __init__(self, index, url_path_list):
                super(DownloadProcess, self).__init__()
                self.index = index
                self.url_path_list = url_path_list
            def run(self):
                print(f"进程{self.index} 开始下载...")
                for url_path in tqdm(self.url_path_list):
                    url = url_path["url"]
                    path = url_path["path"]
                    try:
                        urlretrieve(url, path)
                    except Exception as e:
                        print(f"下载出错------>url={url},path={path}")

        # ------------------------------------------------------------------------
        process_list = []
        url_path_list_list = List_Tools.chunk_N(url_path_list, process_nums)
        for i in range(process_nums):  # 开启N个子进程
            p = DownloadProcess(index=i, url_path_list=url_path_list_list[i])  # 实例化进程对象
            p.start()
            process_list.append(p)
        for i in process_list:
            p.join()
        print("下载结束")
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
