# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/05/25 20:35
# @author  : zh-atom
# @function:

import os
import re
import random
import jieba
import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from conf.path_config import path_train, path_valid, path_label, path_tests, path_category, path_dataset, \
    path_edata, path_embedding_vector_word2vec_word, path_embedding_random_word, path_embedding_vector_word2vec_word_bin, \
    path_embedding_random_char, path_embedding_vector_word2vec_char_bin, path_embedding_vector_word2vec_char, path_embedding_user_dict

str_split = '|,|'

class preprocess_excel_data:
    def __init__(self):
        self.corpus = []
        self.corpus_labels = []
        self.corpus_titles = []

    def removePunctuation(self, content):
        """
        文本去标点
        """
        punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,.，。、《》？；：‘""“”{【】}|、！@#￥%……&*（）——+=- "
        content = re.sub(r'[{}]+'.format(punctuation), '', content)

        if content.startswith(' ') or content.endswith(' '):
            re.sub(r"^(\s+)|(\s+)$", "", content)
        return content.strip()

    def list_all_files(self, rootdir):
        import os
        _files = []
        # 列出文件夹下所有的目录与文件
        list_file = os.listdir(rootdir)

        for i in range(0, len(list_file)):
            # 构造路径
            path = os.path.join(rootdir, list_file[i])
            # 判断路径是否是一个文件目录或者文件
            # 如果是文件目录，继续递归
            if os.path.isdir(path):
                _files.extend(self.list_all_files(path))
            if os.path.isfile(path):
                _files.append(path)
        return _files

    def label_check(self, category, label):
        # 读 category2labels.json文件，校验 类别-标签 是否匹配
        with open(path_category, 'r', encoding='utf-8') as f_c2l:
            c2l_json = json.load(f_c2l)

        if ' ' in label:
            label_tmp = label.split(' ')
            for i in label_tmp:
                if i not in c2l_json[category]:
                    return False

        elif label not in c2l_json[category]:
            return False
        return True

    def gen_jieba_user_dict(self, path):
        lable_dict = []
        for lines in self.corpus_labels:
            for len in lines:
                lable_dict.append(len)
        lable_dict = list(set(lable_dict))
        with open(path, 'w', encoding='utf-8') as f_path:
            for line in lable_dict:
                    f_path.write(line + '\n')
            f_path.close()
        return lable_dict

    def excel2csv(self):
        labels = []
        trains = []
        data = []
        edata =[]
        files = self.list_all_files(os.path.dirname(path_dataset))
        for file in files:
            if file.startswith('0') or file.endswith('.xlsx'):
                print('Will read execel file：' + file)
                data += np.array(pd.read_excel(file)).tolist()

        for s_list in data:
            # print(s_list)
            raw_label = str(s_list[5])
            raw_title = str(s_list[3])
            raw_category = str(s_list[4])
            cov_label = self.removePunctuation(raw_label)
            cov_title = self.removePunctuation(raw_title)
            cov_category = raw_category.strip()

            # 跳过无效数据
            if 'nan' in raw_label or 'nan' in raw_title or 'nan' in raw_category:
                continue
            # 跳过 分类和标签 不匹配的数据
            if self.label_check(cov_category, cov_label) == False:
                edata.append(str(s_list[0]) + str_split + cov_category + str_split + cov_label + str_split + cov_title)
                continue

            label_tmp = cov_label.replace('/', ' ')  # 去除字母标签分类的 ‘/’
            label_tmp = re.sub(r'  ', ' ', label_tmp)  # 去除标签里面的双空格

            # 将 label 和 title 都加入语料库
            self.corpus_labels.append(list(label_tmp.split(' ')))
            # jieba.suggest_freq('十八大', True)    #修改词频，使其不能分离
            # self.corpus.append(list(jieba.cut(cov_title, cut_all=False, HMM=False)))
            self.corpus_titles.append(cov_title)

            # 处理多标签的情况
            if ' ' in label_tmp:
                label_tmp = label_tmp.split(' ')
                train_tmp = []
                for i in label_tmp:
                    labels.append(i)
                    train_tmp.append(i)
                trains.append(','.join(train_tmp) + str_split + cov_title)
            else:
                labels.append(cov_label)
                trains.append(cov_label + str_split + cov_title)

        label_dict = self.gen_jieba_user_dict(path_embedding_user_dict)
        jieba.load_userdict(path_embedding_user_dict)  # 添加自定义词典

        for line in self.corpus_titles:
            self.corpus.append(list(jieba.cut(line, cut_all=False, HMM=False)))
        self.corpus.append(label_dict)

        # 生成 label 文件
        with open(path_label, 'w', encoding='utf-8') as f_label:
            labels = list(set(labels))  # 去重
            labels.sort(reverse=False)  # 排序
            for line in labels:
                f_label.write(line + '\n')
            f_label.close()

        # 生成 train.csv vaild.csv test.csv 文件
        f_train = open(path_train, 'w', encoding='utf-8')
        f_valid = open(path_valid, 'w', encoding='utf-8')
        f_tests = open(path_tests, 'w', encoding='utf-8')
        random.shuffle(trains)
        f_valid.write('label'+ str_split + 'ques' + '\n')
        f_train.write('label'+ str_split + 'ques' + '\n')
        f_tests.write('label'+ str_split + 'ques' + '\n')
        for i in range(len(trains)):
            print(trains[i])
            # 拆分训练集、验证集、测试集
            if i % 5 == 0:
                f_valid.write(trains[i] + '\n')
            #elif i % 11 == 0:
            #    f_tests.write(trains[i] + '\n')
            else:
                f_train.write(trains[i] + '\n')
        f_valid.close()
        f_train.close()
        f_tests.close()

        # 生成有误数据集 error_data.csv 文件
        f_edata = open(path_edata, 'w', encoding='utf-8')
        f_edata.write('province,category,label,ques' + '\n')
        for i in range(len(edata)):
            f_edata.write(edata[i] + '\n')
        f_edata.close()

    def gen_vec(self):
        print(self.corpus)
        word_list = []
        char_list = []
        # 生成 word2vec 预训练 文件
        for line in self.corpus:
            for word in line:
                word_list.append(word)
                for char in word:
                    char_list.append(char)

        word_list = list(set(word_list))
        char_list = list(set(char_list))
        with open(path_embedding_random_word, 'w', encoding='utf-8') as f_word_vec_bin:
            for line in word_list:
                f_word_vec_bin.write(line + '\n')
            f_word_vec_bin.close()

        with open(path_embedding_random_char, 'w', encoding='utf-8') as f_char_vec_bin:
            for line in char_list:
                f_char_vec_bin.write(line + '\n')
            f_char_vec_bin.close()

        print("start to gen word vec file")
        # 嵌入参数： sg=> 0:CBOW 1:SKip-Gram, sentences = word_list
        model_word = Word2Vec(corpus_file=path_embedding_random_word, size=300, window=5, min_count=1, workers=4)
        model_word.wv.save_word2vec_format(path_embedding_vector_word2vec_word_bin, binary=True)
        model_word.wv.save_word2vec_format(path_embedding_vector_word2vec_word, binary=False)

        print("start to gen word vec file")
        model_word = Word2Vec(corpus_file=path_embedding_random_char, size=300, window=5, min_count=1, workers=4)
        model_word.wv.save_word2vec_format(path_embedding_vector_word2vec_char_bin, binary=True)
        model_word.wv.save_word2vec_format(path_embedding_vector_word2vec_char, binary=False)
