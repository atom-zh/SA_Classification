# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 11:29
# @author   :Mo
# @function :embeddings of model, base embedding of random, word2vec or bert

from conf.path_config import path_embedding_vector_word2vec_char, path_embedding_vector_word2vec_word
from conf.path_config import path_embedding_random_char, path_embedding_random_word
from data_preprocess.text_preprocess import get_ngram
from keras.layers import Add, Embedding, Lambda
from gensim.models import KeyedVectors
from keras.models import Input, Model
import numpy as np
import jieba
import os

class BaseEmbedding:
    def __init__(self, hyper_parameters):
        self.len_max = hyper_parameters.get('len_max', 50)  # 文本最大长度, 建议25-50
        self.embed_size = hyper_parameters.get('embed_size', 300)  # 嵌入层尺寸
        self.vocab_size = hyper_parameters.get('vocab_size', 30000)  # 字典大小, 这里随便填的，会根据代码里修改
        self.trainable = hyper_parameters.get('trainable', False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.level_type = hyper_parameters.get('level_type', 'char')  # 还可以填'word'
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')  # 词嵌入方式，可以选择'xlnet'、'bert'、'random'、'word2vec'

        # 自适应, 根据level_type和embedding_type判断corpus_path
        if self.level_type == "word":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_word)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_word)
            elif self.embedding_type == "bert":
                raise RuntimeError("bert level_type is 'char', not 'word'")
            elif self.embedding_type == "xlnet":
                raise RuntimeError("xlnet level_type is 'char', not 'word'")
            elif self.embedding_type == "albert":
                raise RuntimeError("albert level_type is 'char', not 'word'")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "char":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_char)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_char)
            elif self.embedding_type == "bert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_bert)
            elif self.embedding_type == "xlnet":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_xlnet)
            elif self.embedding_type == "albert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_albert)
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "ngram":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path')
                if not self.corpus_path:
                    raise RuntimeError("corpus_path must exists!")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        else:
            raise RuntimeError("level_type must be 'char' or 'word'")
        # 定义的符号
        self.ot_dict = {'[PAD]': 0,
                        '[UNK]': 1,
                        '[BOS]': 2,
                        '[EOS]': 3, }
        self.deal_corpus()
        self.build()

    def deal_corpus(self):  # 处理语料
        pass

    def build(self):
        self.token2idx = {}
        self.idx2token = {}

    def sentence2idx(self, text, second_text=None):
        if second_text:
            second_text = "[SEP]" + str(second_text).upper()
        # text = extract_chinese(str(text).upper())
        text = str(text).upper()

        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index

    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)


class RandomEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.ngram_ns = hyper_parameters['embedding'].get('ngram_ns', [1, 2, 3]) # ngram信息, 根据预料获取
        # self.path = hyper_parameters.get('corpus_path', path_embedding_random_char)
        super().__init__(hyper_parameters)

    def deal_corpus(self):
        token2idx = self.ot_dict.copy()
        count = 3
        if 'term' in self.corpus_path:
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                while True:
                    term_one = fd.readline()
                    if not term_one:
                        break
                    term_one = term_one.strip()
                    if term_one not in token2idx:
                        count = count + 1
                        token2idx[term_one] = count

        elif os.path.exists(self.corpus_path):
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                terms = fd.readlines()
                for term_one in terms:
                    if self.level_type == 'char':
                        text = list(term_one.replace(' ', '').strip())
                    elif self.level_type == 'word':
                        text = list(jieba.cut(term_one, cut_all=False, HMM=False))
                    elif self.level_type == 'ngram':
                        text = get_ngram(term_one, ns=self.ngram_ns)
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word', 'char', 'ngram'")
                    for text_one in text:
                        if text_one not in token2idx:
                            count = count + 1
                            token2idx[text_one] = count
        else:
            raise RuntimeError("your input corpus_path is wrong, it must be 'dict' or 'corpus'")
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        self.input = Input(shape=(self.len_max,), dtype='int32')
        self.output = Embedding(self.vocab_size+1,
                                self.embed_size,
                                input_length=self.len_max,
                                trainable=self.trainable,
                                )(self.input)
        self.model = Model(self.input, self.output)

    def sentence2idx(self, text, second_text=""):
        if second_text:
            second_text = "[SEP]" + str(second_text).upper()
        # text = extract_chinese(str(text).upper()+second_text)
        text =str(text).upper() + second_text
        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=False))
        elif self.level_type == 'ngram':
            text = get_ngram(text, ns=self.ngram_ns)
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        # text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index


class WordEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        # self.path = hyper_parameters.get('corpus_path', path_embedding_vector_word2vec)
        super().__init__(hyper_parameters)

    def build(self, **kwargs):
        self.embedding_type = 'word2vec'
        print("load word2vec start!")
        self.key_vector = KeyedVectors.load_word2vec_format(self.corpus_path, **kwargs)
        print("load word2vec end!")
        self.embed_size = self.key_vector.vector_size

        self.token2idx = self.ot_dict.copy()
        embedding_matrix = []
        # 首先加self.token2idx中的四个[PAD]、[UNK]、[BOS]、[EOS]
        embedding_matrix.append(np.zeros(self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))

        for word in self.key_vector.index2entity:
            self.token2idx[word] = len(self.token2idx)
            embedding_matrix.append(self.key_vector[word])

        # self.token2idx = self.token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

        self.vocab_size = len(self.token2idx)
        embedding_matrix = np.array(embedding_matrix)
        self.input = Input(shape=(self.len_max,), dtype='int32')

        self.output = Embedding(self.vocab_size,
                                self.embed_size,
                                input_length=self.len_max,
                                weights=[embedding_matrix],
                                trainable=self.trainable)(self.input)
        self.model = Model(self.input, self.output)
