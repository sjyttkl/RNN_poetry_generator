# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     Data.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/4/26
   Description :  
==================================================
"""
__author__ = 'songdongdong'


import numpy as np
import collections
from  config import Config


class Data:
    def __init__(self):
        self.batch_size = Config.batch_size
        self.poetry_file = Config.poetry_file
        self.load()
        self.create_batches()

    def load(self):
        def handle(line): #返回 开始字符^ + 诗句+ 结束字符$
            if len(line) > Config.MAX_LENGTH:#诗句字数最大长度
                index_end = line.rfind('。', 0, Config.MAX_LENGTH)#诗句字数最大长度，超过就不要了，进行截断
                index_end = index_end if index_end > 0 else Config.MAX_LENGTH #确定诗句最后一个位置的索引
                line = line[:index_end + 1] #返回诗句最大长度内的数据
            return Config.BEGIN_CHAR + line + Config.END_CHAR

        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        open(self.poetry_file, encoding='utf-8')]
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > Config.MIN_LENGTH]

        words = []# 存储所有字
        for poetry in self.poetrys: #存储所有 的  ^ +诗句 + $
            words += [word for word in poetry]
        counter = collections.Counter(words) #映射成 字典类型：key(字符):value(词频)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1]) #按照  第二位（词频）进行 降序排列
        words, _ = zip(*count_pairs) #取出所有词语（词频高的在前面）
        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(Config.max_words, len(words)) #出现的字的数量限制了下，3000个
        self.words = words[:words_size] + (Config.UNKNOWN_CHAR,) #加上了一个未登录词
        self.words_size = len(self.words)
        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(Config.UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys] #map函数的使用，把所有poetry应用到 lambda(相当于一个方法)中

    def create_batches(self):
        self.n_size = len(self.poetrys_vector) // self.batch_size  #每次  需要训练的次数
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            length = max(map(len, batches)) #batches里长度最长诗句的长度。
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r  #字符数不够长的，则进行补充 未登录词的id
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:] #ydata这里剔除了前后 两个字符
            self.x_batches.append(xdata) #以batch为单位，存在x_batches
            self.y_batches.append(ydata)#以batch为单位，存在y_batches