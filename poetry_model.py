# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     poetry_model.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/4/26
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
# import tensorflow.contrib.rnn as rnn
import tensorflow.nn.rnn_cell as rnn_cell  #新路径
import tensorflow.contrib.legacy_seq2seq as seq2seq
from  config import Config
class Model:
    def __init__(self, data,  infer=False):
        self.rnn_size = Config.run_size
        self.n_layers = Config.n_layers
        self.model = Config.model

        if infer: #预测，生成
            self.batch_size = 1
        else: #训练
            self.batch_size = data.batch_size

        if self.model == 'rnn':
            cell_rnn = rnn_cell.BasicRNNCell
        elif self.model == 'gru':
            cell_rnn = rnn_cell.GRUCell
        elif self.model == 'lstm':
            cell_rnn = rnn_cell.BasicLSTMCell
        else:
            cell_rnn = rnn_cell.LSTMCell
        cell = cell_rnn(self.rnn_size, state_is_tuple=True)

        self.multi_cell  = rnn_cell.MultiRNNCell([cell] * self.n_layers, state_is_tuple=True) #这里建议改为True

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])

        self.initial_state = self.multi_cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [data.words_size, self.rnn_size])

                # 将原本单词ID转为单词向量。
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf) # #batch_size , num_steps ,HIDDEN_SIZE

        outputs, final_state = tf.nn.dynamic_rnn(
            self.multi_cell, inputs, initial_state=self.initial_state, scope='rnnlm')


        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],#展开成一维【列表】
                                                [tf.ones_like(pred, dtype=tf.float32)] #权重
                                                )

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(1.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))