# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of RCNNGraph_kim with baidu-qa-2019 in question title


# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 训练验证数据地址
from conf.path_config import path_train, path_valid, path_tests, path_out
# 数据预处理, 删除文件目录下文件
from data_preprocess.text_preprocess import PreprocessTextMulti, delete_file, load_json, save_json, transform_multilabel_to_multihot, txt_write
from data_preprocess.utils import mkdir, draw_accuracy_figure
# 模型图
from mTextRCNN.graph import RCNNGraph as Graph
# 模型评估
from sklearn.metrics import classification_report, hamming_loss
# 计算时间
import time

def train(hyper_parameters=None, rate=1.0):
    """
        训练函数
    :param hyper_parameters: json, 超参数
    :param rate: 比率, 抽出rate比率语料取训练
    :return: None
    """
    if not hyper_parameters:
        hyper_parameters = {
        'train_name':'textRNN',
        'train_mode':'non', # non, attention
        'train_time': None,
        'path_train_out': 'None',
        'len_max': 60,  # 句子最大长度, 固定 推荐20-50
        'embed_size': 300,  # 字/词向量维度
        'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
        'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
        'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word'
        'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
        'gpu_memory_fraction': 0.86, #gpu使用率
        'model': {'label': 51,  # 类别数
                  'batch_size': 256,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                  'filters': [2, 3, 4, 5],  # 卷积核尺寸
                  'filters_num': 300,  # 卷积个数 text-cnn:300-600
                  'channel_size': 1,  # CNN通道数
                  'dropout': 0.2,  # 随机失活, 概率
                  'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
                  'decay_rate': 0.9,  # 学习率衰减系数, 乘法
                  'epochs': 1,  # 训练最大轮次
                  'patience': 3, # 早停,2-3就好
                  'lr': 5e-4,  # 学习率, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                  'l2': 1e-9,  # l2正则化
                  'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数
                  #'loss': 'categorical_crossentropy',  # 损失函数
                  'loss': 'binary_crossentropy',  # 损失函数, 可能有问题, 可以自己定义 categorical_crossentropy
                  'metrics': 'accuracy',  # 保存更好模型的评价标准
                  'is_training': True,  # 训练后者是测试模型
                  'path_model_dir': path_model_dir,  # 模型目录
                  'model_path': path_model,
                  # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                  'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                  'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                  'rnn_type': 'LSTM',  # rnn_type类型, 还可以填"GRU"
                  'rnn_units': 256,    # RNN隐藏层
                  },
        'embedding': {'layer_indexes': [12], # bert取的层数,
                      # 'corpus_path': '',     # embedding预训练数据地址,不配则会默认取conf里边默认的地址, keras-bert可以加载谷歌版bert,百度版ernie(需转换，https://github.com/ArthurRizar/tensorflow_ernie),哈工大版bert-wwm(tf框架，https://github.com/ymcui/Chinese-BERT-wwm)
                        },
        'data':{'train_data': path_train,  # 训练数据
                'val_data': path_valid,    # 验证数据
                'test_data': path_tests,  # 测试数据
                },
    }

    output_path = path_out + hyper_parameters['train_name'] + '_' + hyper_parameters['train_mode']
    mkdir(output_path)
    hyper_parameters['path_train_out'] = output_path

    # 删除先前存在的模型\embedding微调模型等
    delete_file(path_model_dir)
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessTextMulti(path_model_dir)
    x_train, y_train = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                       hyper_parameters['data']['train_data'],
                                                       ra_ed, rate=rate, shuffle=True)
    x_val, y_val = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                   hyper_parameters['data']['val_data'],
                                                   ra_ed, rate=rate, shuffle=True)
    print("data propress ok!")
    print(len(y_train))
    # 训练
    H = graph.fit(x_train, y_train, x_val, y_val)

    train_time = time.time()-time_start
    hyper_parameters['train_time'] = train_time
    print("耗时:" + str(train_time))

    # 绘图
    draw_accuracy_figure(H, output_path)

def pred_tet(path_hyper_parameter=path_hyper_parameters, path_test=None, rate=1.0):
    preout_parameters = {
        'predict_time': None,
        'predict_acc': None,
        'hamming_loss': None,
        'predict_report': 'None',
    }
    # 测试集的准确率
    hyper_parameters = load_json(path_hyper_parameter)
    path_json = hyper_parameters['path_train_out'] + '/predict_resault.json'
    time_start = time.time()

    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    graph.load_model()
    print("graph load ok!")
    ra_ed = graph.word_embedding

    # 数据预处理
    pt = PreprocessTextMulti(path_model_dir)
    x, y = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'], path_test,
                                                                ra_ed, rate, shuffle=True)
    y_pred = []
    index_y = []
    pred = graph.predict(x)

    print(pred)
    for i in range(len(pred)):
        pre = pt.prereocess_idx(pred[i])
        label_pred = pre[0][0][0]
        label_pred = pt.l2i_i2l['l2i'][label_pred]
        label_multi_idex = transform_multilabel_to_multihot(label_pred, label=51)
        y_pred.append(label_multi_idex)
        index_y.append(y[i].tolist())
        print(pre)
        print(label_multi_idex)
        print(y[i].tolist())
        print('=========================')

    print("data pred ok!")
    # 预测结果转为int类型
    #index_y = [pt.l2i_i2l['l2i'][i] for i in y]
    #index_pred = [pt.l2i_i2l['l2i'][i] for i in y_pred]
    target_names = [pt.l2i_i2l['i2l'][str(i)] for i in range(hyper_parameters['model'].get('label', '51'))]
    print(target_names)
    # 评估
    report_predict = classification_report(index_y, y_pred, digits=9, target_names=target_names)
    preout_parameters['predict_report'] = report_predict
    print(report_predict)
    txt_write(list_line=report_predict, file_path=hyper_parameters['path_train_out'] + '/report.txt')

    h_loss = hamming_loss(index_y, y_pred)
    preout_parameters['hamming_loss'] = h_loss
    print("Hamming Loss = {:.6f}".format(h_loss))

    predict_time = time.time() - time_start
    preout_parameters['predict_time'] = predict_time
    print("耗时:" + str(predict_time))

    save_json(jsons=preout_parameters, json_path=path_json)

if __name__=="__main__":
    train(rate=1)
    pred_tet(path_test=path_valid, rate=1) # sample条件下设为1,否则训练语料可能会很少