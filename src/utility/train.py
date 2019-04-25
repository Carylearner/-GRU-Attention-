"""
训练模块包括训练的过程定义、评估的方法定义
"""

import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from utility.common import logger
from utility.process import SOS_token,MAX_LENGTH
from torch import nn
from torch import optim
from torch.autograd import Variable
from utility.translation import variableFromSentence,variablesFromPair
from utility.translation import batch2TrainData
from utility.process import EOS_token,SOS_token
from utility.function import maskNLLLoss,timeSince

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
teacher_forcing_ratio = 0.5

def showPlot(points):
    """
    绘制图像
    :param points:
    :return:
    """
    plt.figure()
    fig,ax = plt.subplots()
    # 绘图间隔设置
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(input_lang,output_lang,encoder,decoder,sentence,max_length=MAX_LENGTH):
    """
    单句评估
    :param input_lang:  原语言信息
    :param output_lang:  目标语言信息
    :param encoder:  编码器
    :param decoder:  解码器
    :param sentence: 要评估的句子
    :param max_length: 可接受最大长度
    :return: 翻译过的句子和注意力信息
    """
    # 输入句子预处理
    input_variable = variableFromSentence(input_lang,sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length,encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output,encoder_hidden = encoder(input_variable[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei]+encoder_output[0][0]
    decoder_input = Variable(torch.Tensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    # 翻译过程
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        # 当前时刻输出为句子结束标志，则结束
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


teacher_forcing_ratio = 0.5



def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_embedding,decoder_embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    '''
    单次训练过程，
    :param input_variable: 源语言信息
    :param target_variable: 目标语言信息
    :param encoder: 编码器
    :param decoder: 解码器
    :param encoder_optimizer: 编码器的优化器
    :param decoder_optimizer: 解码器的优化器
    :param criterion: 评价准则，即损失函数的定义
    :param max_length: 接受的单句最大长度
    :return: 本次训练的平均损失
    '''
    encoder_hidden = encoder.initHidden()

    # 清楚优化器状态
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 编码过程
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    # print(input_length, " -> ", target_length)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    # print("encoder_outputs shape ", encoder_outputs.shape)
    loss = 0
    print_losses=[]
    n_totals = 0


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs  # 这里decoder_output的size是64x单词的个数
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal


    # 反向传播
    loss.backward()
    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)   #修剪梯度，防止梯度爆炸
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 网络状态更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    '''
    训练过程,可以指定迭代次数，每次迭代调用 前面定义的train函数，并在迭代结束调用绘制图像的函数
    :param input_lang: 输入语言实例
    :param output_lang: 输出语言实例
    :param pairs: 语料中的源语言-目标语言对
    :param encoder: 编码器
    :param decoder: 解码器
    :param n_iters: 迭代次数
    :param print_every: 打印loss间隔
    :param plot_every: 绘制图像间隔
    :param learning_rate: 学习率
    :return:
    '''
    start = time.time()
    batch_size = 50
    plot_losses = []
    clip = 50
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    # Zero gradients
    training_batches = [batch2TrainData(input_lang,output_lang, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iters)]
    # 损失函数定义
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        logger.info("第{}次循环".format(iter))
        training_batch = training_batches[iter]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss =train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                               iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    '''
    从语料中随机选取句子进行评估
    '''
    for i in range(n):
        pair = random.choice(pairs)
        logger.info('> %s' % pair[0])
        logger.info('= %s' % pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        logger.info('< %s' % output_sentence)
        logger.info('')



def showAttention(input_sentence, output_words, attentions):
    try:
        # 添加绘图中的中文显示
        plt.rcParams['font.sans-serif'] = ['STSong']  # 宋体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 使用 colorbar 初始化绘图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # 设置x，y轴信息
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # 显示标签
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
    except Exception as err:
        logger.error(err)


def evaluateAndShowAtten(input_lang, ouput_lang, input_sentence, encoder1, attn_decoder1):
    output_words, attentions = evaluate(input_lang, ouput_lang,
                                        encoder1, attn_decoder1, input_sentence)
    logger.info('input = %s' % input_sentence)
    logger.info('output = %s' % ' '.join(output_words))
    # 如果是中文需要分词
    if input_lang.name == 'cmn':
        print(input_lang.name)
       # input_sentence = cut(input_sentence)
    showAttention(input_sentence, output_words, attentions)


