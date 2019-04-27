# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     config.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/4/26
   Description :  配置文件
==================================================
"""
__author__ = 'songdongdong'

class Config(object):
    BEGIN_CHAR = '^'
    END_CHAR = '$'
    UNKNOWN_CHAR = '*'
    MAX_LENGTH = 100 #诗句最大长度
    MIN_LENGTH = 10
    max_words = 3000 #出现最多的字 最多有 3000个，
    epochs = 50
    batch_size = 64 #每次需要训练的数据量
    poetry_file = 'poetry.txt' #训练数据
    save_dir = 'save_model' #模型保存路径
    run_size = 128 #隐藏节点
    n_layers = 2 #层数
    model = "lstm"