"""
 这个是数据处理模块，内容包括从文件加载数据、数据解析和一些辅助函数
"""
#

import math,re,time,unicodedata,jieba,torch
from utility.common import logger
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
# 中文的时候要设置大一些
MAX_LENGTH = 25
eng_prefixes = ("i am","i m","he is","he s","she is","she s",
                "you are","you re","we are","we re","they are","they re")

