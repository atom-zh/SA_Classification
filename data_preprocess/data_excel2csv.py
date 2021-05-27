# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/05/25 20:35
# @author  : zh-atom
# @function:

from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json, txt_read
from keras_textclassification.conf.path_config import path_model_dir
from keras_textclassification.conf.path_config import path_train, path_valid, path_label, path_root
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import re

def removePunctuation(content):
    """
    文本去标点
    """
    punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,.，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"
    content = re.sub(r'[{}]+'.format(punctuation), '', content)

    if content.startswith(' ') or content.endswith(' '):
        re.sub(r"^(\s+)|(\s+)$", "", content)
    return content.strip()

def excel2csv():
    labels = []
    trains = ['label|,|ques']
    data = pd.read_excel(os.path.dirname(path_train)+'/02-anhui.xlsx')
    data = np.array(data)
    data = data.tolist()
    for s_list in data:
        print(s_list)
        label_tmp = removePunctuation(s_list[5])
        if ' ' in label_tmp:
            train_tmp = []
            label_tmp = label_tmp.split(' ')
            for i in label_tmp:
                label = removePunctuation(s_list[4]) + '/' + removePunctuation(i)
                labels.append(label)
                train_tmp.append(label)
            train = ','.join(train_tmp) + '|,|' + removePunctuation(s_list[3])
            trains.append(train)
        else:
            label = removePunctuation(s_list[4]) + '/' + removePunctuation(s_list[5])
            labels.append(label)
            trains.append(label + '|,|' + removePunctuation(s_list[3]))

    # 生成 label 文件
    with open(path_label, 'w', encoding='utf-8') as f_label:
        labels = list(set(labels))
        labels.sort(reverse=False)
        for line in labels:
            f_label.write(line + '\n')
        f_label.close()

    # 生成 train.csv 文件
    with open(path_train, 'w', encoding='utf-8') as f_train:
        for line in trains:
            f_train.write(line + '\n')
        f_train.close()

    return None
