"""
这个公共模块主要是日志的处理。在数据处理、模型训练等过程中需要保留需要的日志，
这样可以对程序的运行过程、运行结果进行记录和分析。这里定义了logger日志对象

"""

import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S-',
                   filename='log.log',
                   filemode='a')


#设置日志的控制台输出
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-6s: %(levelname)-6s %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)


