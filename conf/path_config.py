# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:04
# @author   :Mo
# @function :file of path

import os

# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_root = path_root.replace('\\', '/')

# train out
path_out = path_root + "/out/"

# path of embedding
path_embedding = path_out + 'data/embeddings'
path_embedding_user_dict = path_embedding + '/user_dict.txt'
path_embedding_random_char = path_embedding + '/term_char.txt'
path_embedding_random_word = path_embedding + '/term_word.txt'
path_embedding_vector_word2vec_char = path_embedding + '/multi_label_char.vec'
path_embedding_vector_word2vec_word = path_embedding + '/multi_label_word.vec'
path_embedding_vector_word2vec_char_bin = path_embedding + '/multi_label_char.bin'
path_embedding_vector_word2vec_word_bin = path_embedding + '/multi_label_word.bin'

path_dataset = path_root +'/dataset'
path_category = path_dataset + '/category2labels.json'
path_l2i_i2l = path_dataset + '/l2i_i2l.json'

# classfiy multi labels 2021
path_multi_label = path_out + 'data/multi_label'
path_multi_label_train = path_multi_label + '/train.csv'
path_multi_label_valid = path_multi_label + '/valid.csv'
path_multi_label_labels = path_multi_label + '/labels.csv'
path_multi_label_tests = path_multi_label + '/tests.csv'
path_multi_label_error = path_multi_label + '/error.csv'

# 路径抽象层
path_label = path_multi_label_labels
path_train = path_multi_label_train
path_valid = path_multi_label_valid
path_tests = path_multi_label_tests
path_edata = path_multi_label_error

# 模型目录
path_model_dir =  path_out + "data/model"
# 语料地址
path_model = path_model_dir + '/model_fast_text.h5'
# 超参数保存地址
path_hyper_parameters =  path_model_dir + '/hyper_parameters.json'
# embedding微调保存地址
path_fineture = path_model_dir + "/embedding_trainable.h5"
