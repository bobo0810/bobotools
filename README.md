# bobotools
[![OSCS Status](https://www.oscs1024.com/platform/badge/bobo0810/bobotools.svg?size=small)](https://www.oscs1024.com/project/bobo0810/bobotools?ref=badge_small)
- 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

## 安装

```bash
pip install bobotools
or
pip install git+https://github.com/bobo0810/bobotools.git
```

## 工具类

### Pytorch(torch_tools)
|  方法   | 功能  |
|  ----  | ----  |
| get_model_info  | 获取模型信息(模型大小、计算量、参数量、前向推理耗时等) |
| vis_tensor  | 可视化tensor|
| vis_cam  | 可视化注意力图|


### 图像(img_tools)
|  方法   | 功能  |
|  ----  | ----  |
| download_url  | 多进程下载URL图像|
| filter_md5  | 对输入的图像列表，互相去重，返回重复图像列表|
| verify_integrity  | 对输入的图像列表，验证完整性，返回错误图像列表|
| plot_yolo  | 可视化yolo结果|




### 文本(txt_tools)
|  方法   | 功能  |
|  ----  | ----  |
| read_lines  | 批量读取txt，支持指定分隔符 |
| write_lines  | 批量写入,保存为txt |

### 列表(list_tools)
|  方法   | 功能  |
|  ----  | ----  |
| chunk_N  | 列表均分为N块 |
| chunk_per| 列表分块，每块为指定长度M |


## 参考

- [Python打包](https://www.jianshu.com/p/9a5e7c935273)


