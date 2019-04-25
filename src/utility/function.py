"""
这里主要定义一些辅助函数，这样可以简化程序的逻辑，能够复用
"""
import unicodedata
import re
import jieba
from utility.process import MAX_LENGTH,eng_prefixes
import math
import time
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r"\1",s)
    return s

def cut(sentence,use_jieba=False):
    """
    对句子进行分词
    :param sentence: 要分词的句子
    :param use_jieba: 是否使用jieba进行智能分词，默认按单字切分
    :return: 分词结果，空格区分
    """
    if use_jieba:
        return ' '.join(jieba.cut(sentence))
    else:
        words = [word for word in sentence]
    return ' '.join(words)

def filterPair(p):
    """
    按自定义最大长度过滤
    :param p:
    :return:
    """
    return len(p[0].split(' '))<MAX_LENGTH and len(p[1].split(' '))< MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '{:d}m {:d}s'.format(m,s)

def timeSince(since,percent):
    now = time.time()
    s = now-since
    es = s/(percent)
    rs = es-s
    return '{:s}(s{:s}'.format(asMinutes(s),asMinutes(rs))


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)) #这个方法真是绕,其中target必须是一个longTensor
    loss = crossEntropy.masked_select(mask).mean()  #其中mask必须是一个ByteTensor,另外这个函数是真好用,通过0\1矩阵就可以轻易处理句子长短不同的情况
    loss = loss.to(device)
    return loss, nTotal.item()