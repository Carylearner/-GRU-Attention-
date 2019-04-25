# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("TkAgg")
import pickle
import sys
from io import open

import torch
from utility.common import logger
from utility.model import LuongAttnDecoderRNN
from utility.model import EncoderRNN
from utility.translation import prepareData
from utility.train import *
import torch




use_cuda = torch.cuda.is_available()
logger.info("Use cuda:{}".format(use_cuda))
input = 'eng'
output = 'cmn'
# 从参数接收要翻译的语种名词
if len(sys.argv) > 1:
    output = sys.argv[1]
logger.info('%s -> %s' % (input, output))

# 处理语料库
input_lang, output_lang, pairs = prepareData(input, output, True)
logger.info(random.choice(pairs))

# 查看两种语言的词汇大小情况
logger.info('input_lang.n_words: %d' % input_lang.n_words)
logger.info('output_lang.n_words: %d' % output_lang.n_words)

# 保存处理过的语言信息，评估时加载使用
pickle.dump(input_lang, open('./data/%s_%s_input_lang.pkl' % (input, output), "wb"))
pickle.dump(output_lang, open('./data/%s_%s_output_lang.pkl' % (input, output), "wb"))
pickle.dump(pairs, open('./data/%s_%s_pairs.pkl' % (input, output), "wb"))
logger.info('lang saved.')

# 编码器和解码器的实例化
model_name = 'cb_model'
attn_model = 'dot'
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
hidden_size = 500
encoder_embedding = nn.Embedding(input_lang.n_words,hidden_size)
decoder_embedding = nn.Embedding(output_lang.n_words,hidden_size)
encoder1 = EncoderRNN(hidden_size,encoder_embedding,encoder_n_layers)
attn_decoder1 = LuongAttnDecoderRNN(attn_model,decoder_embedding,hidden_size,output_lang.n_words,decoder_n_layers)
if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

logger.info('train start. ')
# 训练过程，指定迭代次数，此处为迭代75000次，每5000次打印中间信息
trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 75000, print_every=5000)
logger.info('train end. ')

# 保存编码器和解码器网络状态
torch.save(encoder1.state_dict(), open('./data/%s_%s_encoder1.stat' % (input, output), 'wb'))
torch.save(attn_decoder1.state_dict(), open('./data/%s_%s_attn_decoder1.stat' % (input, output), 'wb'))
logger.info('stat saved.')

# 保存整个网络
torch.save(encoder1, open('./data/%s_%s_encoder1.model' % (input, output), 'wb'))
torch.save(attn_decoder1, open('./data/%s_%s_attn_decoder1.model' % (input, output), 'wb'))
logger.info('model saved.')
