"""
 对原语言和目标语言进行建模，保存语言相关的所有词的信息，可以在后续的训练和评估中使用
"""
from utility.function import cut,normalizeString,filterPairs
from utility.common import logger
from utility.process import EOS_token,use_cuda
from torch.autograd import Variable
import torch
import itertools



class Lang:
    def __init__(self,name):
        """
        添加need_cut可根据雨中进行不同的切分逻辑处理
        :param name: 语种名称
        """
        self.name = name
        self.need_cut = self.name=='cmn'
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2 #初始化词数为2

    def addSentence(self,sentence):
        """
        从语料中添加句子到Lang
        :param sentence: 语料中的每个句子
        :return:
        """
        if self.need_cut:
            sentence = cut(sentence,True)
        for word in sentence.split(' '):
            if len(word)>0:
                self.addWord(word)

    def addWord(self,word):
        """
        向Lang中添加每个词，并统计词频，如果是新词修改词表大小
        :param word:
        :return:
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1,lang2,reverse=False):
    """
    :param lang1: 原语言
    :param lang2: 目标语言
    :param reverse: 是否逆向翻译
    :return: 原语言实例，目标语言实例，词语对
    """
    logger.info("Reading lines...")
    # 读取txt文件并分割成行
    lines = open('./data/{:s}-{:s}.txt'.format(lang1,lang2),encoding='utf-8').read().strip().split('\n')
    # 按行处理原语言-目标语言对，并做预处理
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 生成语言实例
    if reverse:
        pairs = [list(reversed(p)) for p in pairs ]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang,output_lang,pairs

def prepareData(lang1,lang2,reverse=False):
    input_lang,output_lang,pairs = readLangs(lang1,lang2,reverse)
    logger.info("Read {:d} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    logger.info("Trimmed to {:d} sentence pairs".format(len(pairs)))
    logger.info("Counting words....")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logger.info("Counted words:")
    logger.info("{:s},{:d}".format(input_lang.name,input_lang.n_words))
    logger.info("{:s},{:d}".format(output_lang.name,output_lang.n_words))
    return input_lang,output_lang,pairs

def indexesFromSentence(lang,sentence):
    if lang.need_cut:
        sentence = cut(sentence,True)
    return [lang.word2index[word] for word in sentence.split(' ') if len(word)>0] + [EOS_token]

def variableFromSentence(lang,sentence):
    """
    将指定的句子转换成Variable
    :param lang:
    :param sentence:
    :return:
    """
    if lang.need_cut:
        sentence = cut(sentence)
    indexes = indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(input_lang,output_lang,pair):
    input_variable = variableFromSentence(input_lang,pair[0])
    target_variable = variableFromSentence(output_lang,pair[1])
    return (input_variable,target_variable)



def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(input_lang,output_lang, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, input_lang)
    output, mask, max_target_len = outputVar(output_batch, output_lang)
    return inp, lengths, output, mask, max_target_len


