import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

def draw_accuracy_figure(H, output_path):
    # H: history
    # 准确率图形输出
    N = np.arange(0, H.epoch.__len__())
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Training Accuracy (Multi Labels)")
    plt.plot(N, H.history['acc'], 'bo-', label='train')
    plt.plot(N, H.history['val_acc'], 'r^:', label='test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Training loss (Multi Labels)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(N, H.history['loss'], 'bo-', label='train_loss')
    plt.plot(N, H.history['val_loss'], 'r^:', label='test_loss')
    plt.legend()
    mkdir(output_path)
    plt.savefig(output_path + '/train')

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def copy_file(source, target):
    try:
        copyfile(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:")
    print("\nFile copy done!\n")