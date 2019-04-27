# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     trian
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/4/26
   Description :  
==================================================
"""
__author__ = 'songdongdong'


import tensorflow as tf
import argparse
import sys
import os
import time
import numpy as np
from  poetry_model import Model
from data import Data
from config import Config



def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(Config.save_dir)
        # if model_file!=None:
        #     saver.restore(sess, model_file)
        n = 0
        for epoch in range(Config.epochs): #一共循环多少次
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch))) #赋值
            pointer = 0
            for batch in range(data.n_size): #每次训练的数据量
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batch,
                            Config.epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                # save
                if (epoch * data.n_size + batch) % 1000 == 0 \
                        or (epoch == Config.epochs-1 and batch == data.n_size-1):
                    checkpoint_path = os.path.join(Config.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')


def sample(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(Config.save_dir)
        # print(model_file)
        if model_file != None:
            saver.restore(sess, model_file)
        elif model_file == None:
            raise Exception("没有模型加载,请先训练")
        if head:
            print('生成藏头诗 ---> ', head)
            poem = Config.BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))]) #映射成 ^ + 字
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = Config.BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != Config.END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():
    msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sample',
                        help=u'usage: train or sample, sample is default')
    parser.add_argument('--head', type=str, default='',
                        help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'sample':
        infer = True  # True
        data = Data()
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data()
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()