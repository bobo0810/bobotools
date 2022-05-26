import math
class List_Tools(object):
    '''
    List操作
    '''
    def __init__(self):
        pass

    @staticmethod
    def chunk_per(data_list,per_nums):
        '''
        list分为若干块,块内长度尽量为per_nums
        '''
        assert len(data_list)>0 and per_nums>0
        return [data_list[i:i+per_nums] for i in range(0, len(data_list), per_nums)]
    
    @staticmethod
    def chunk_N(data_list,chunk_nums):
        '''
        list尽量均分块,块数为chunk_nums
        '''
        n = int(math.ceil(len(data_list) / float(chunk_nums)))
        return [data_list[i:i + n] for i in range(0, len(data_list), n)]
