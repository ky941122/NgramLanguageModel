# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author: yadi Lao
import os
import re
import codecs
import pickle
import time
import logging
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE


def load_corpus(corpus_path):
    """
    载入语料
    """
    corpus = []
    with codecs.open(corpus_path, 'r') as f:
        content = f.readlines()
        for lines in content:
            corpus.append(list(lines.replace('\n', '')))

    print('finish loading corpus, size={}'.format(len(corpus)))
    return corpus


def clean(line):
    """
    去掉标点符号
    """
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".encode('utf-8').decode('utf-8'),
                        "".encode('utf-8').decode('utf-8'), line)
    return string


class StatisticNGram_CHAR():
    """
    基于统计的NGram
    """
    def __init__(self, n=3):
        self.n = n
        self.model = None

    def train_model(self, corpus, n):
        """
        MLE训练基于统计的语言模型
        :param text: [['a','b'],['a','b','c']]
        """
        train_data, padded_sents = padded_everygram_pipeline(n, corpus)
        # train model
        t1 = time.time()
        self.model = MLE(n)
        self.model.fit(train_data, padded_sents)
        print('training LM takes {} time'.format(time.time()-t1))

    def calculate_marginal_prob(self, word):
        """
        返回词的边缘概率
        """
        return self.model.score(word)

    def calculate_conditional_prob(self, word, context):
        """
        返回词的条件概率, context的形式为['a','b']
        """
        if len(context) > self.n -1:
            logging.error('[conditional prob]: size of context is larger than ngram setting!!')
            return -1
        else:
            return self.model.score(word, context)

    def restore_model(self, model_path):
        """
        载入语言模型
        """
        print('restore model from {}'.format(model_path))
        f = open(model_path, 'rb')
        self.model = pickle.load(f)
        self.n = pickle.load(f)
        print('ngram={}'.format(self.n))

    def save_model(self, model_path):
        """
        保存模型
        """
        f = open(model_path, 'wb')
        pickle.dump(self.model, f)
        pickle.dump(self.n, f)

    def output_prob_for_excel(self, excel, num):
        """
        解析Excel，并返回加入了prob的dataFrame
        """
        if num < 0 or num > self.n-1:
            logging.error('size of context is larger than ngram setting!!')
            return None
        else:
            df = pd.read_excel(excel)
            print(df.shape)
            # 清洗，分词，填初始标签
            data = np.array(df)
            data = [list(pad_both_ends(str(line), n=2)) for line in data[:,3]]
            print(len(data), type(data))
            print(data[0], type(data[0]))

            if num == 0:
                prob_list = []
                for line in data:
                    prob_dict = []
                    for word in line:
                        prob_dict.append(math.log(self.calculate_marginal_prob(word)+1e-8))
                    prob_list.append(sum(prob_dict)/len(prob_dict))

            else:
                prob_list = []
                for line in data:
                    prob_dict = []
                    for i in range(len(line)-num):
                        context = line[i:i+num]
                        word = line[i+num]
                        prob = self.calculate_conditional_prob(word, context)
                        # prob_dict.append((word+'|'+','.join(context), prob))
                        prob_dict.append(math.log(prob + 1e-8))
                    prob_list.append(sum(prob_dict)/len(prob_dict))

            # 更新df
            df['prob'] = prob_list
            df.to_excel(excel)

            return df


if __name__ == '__main__':
    corpus_path = '../data/label_corpus_extend.txt'   # corpus used to train LM
    model_path = '../data/lm_model_char.pkl'          # path to save and reload LM model
    is_train = False
    n = 3   # maximum ngram

    # num range: [0, n)
    # num=0 --> marginal prob for single word.
    # num>0 --> conditional prob

    num = 0

    if is_train:
        ngram = StatisticNGram_CHAR()
        corpus = load_corpus(corpus_path)
        ngram.train_model(corpus=corpus, n=n)
        ngram.save_model(model_path=model_path)
    else:
        ngram = StatisticNGram_CHAR()
        ngram.restore_model(model_path=model_path)
        data_folder = '/Users/faye/Documents/红线词标注任务/labeled/'    # excel 列表
        file_list = os.listdir(data_folder)
        file_list = list(map(lambda x: ''.join([data_folder, x]), file_list))

        count = 0
        for file in file_list:
            print('process {} file={} '.format(count, file))
            df = ngram.output_prob_for_excel(file, num=num)
            count += 1


