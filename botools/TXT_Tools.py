import os
class TXT_Tools(object):
    '''
    Txt操作
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def read_lines(txt_path,split_flag=None):
        '''
        批量读取多行
        txt_path: txt路径
        split_flag: 分隔符(可选),每行按指定字符分割
        '''
        assert ".txt" in txt_path
        try:
            # 判断文件是否存在
            if not os.path.exists(txt_path):
                return "txt_file not exists"
            else:
                f = open (txt_path,'r')
                lines = f.readlines()
                f.close()

                new_lines=[]
                for line in lines:
                    new_lines.append(line.strip() if split_flag is None else line.strip().split(split_flag))
                return new_lines
        except:
            return "txt_file read error"
    
    @staticmethod
    def write_lines(lines,txt_path):
        '''
        批量写入多行
        lines: 待写入的list
        txt_path: txt保存路径
        '''
        assert ".txt" in txt_path
        # 判断目录
        if not os.path.exists(os.path.dirname(txt_path)):
            os.makedirs(os.path.dirname(txt_path))
            print("create a directory ",os.path.dirname(txt_path))
        # 写入
        file = open(txt_path,'w')
        for line in lines:
            file.write(str(line)+"\n")
        file.close()
        print("generate txt success: ",txt_path)



