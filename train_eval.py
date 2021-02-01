# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def train_test(config, model, train_loader, dev_loader, test_loader):
    start = time.time()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减  每个epoch: lr = gamma*lr
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    create_dir_not_exists(config.model_save_dir)
    create_dir_not_exists(config.plot_save_dir)
    if os.path.exists(config.model_save_name):  # 存在则加载模型 并继续训练
        ckpt = torch.load(config.model_save_name)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        max_acc = ckpt['dev_acc']
        dev_acc_ls = ckpt['dev_acc_ls']
        best_epoch = start_epoch
        print(f'Load epoch {start_epoch} successful...')
    else:
        start_epoch = 0
        max_acc = float('-inf')  # 负无穷
        dev_acc_ls = []
        best_epoch = start_epoch
        print('Start training on epoch 0')
    for epoch in range(start_epoch + 1, config.num_epoch + 1):
        # train and evaluate
        for i, data in enumerate(train_loader):
            model.train()  # 开启dropout和BatchNormal
            x, labels = data  # x[0] - 句子对应索引组成的数组 x[1] - 句子长度组成的数组
            predict = model(x)  # forward
            loss = criterion(predict, labels)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # backward
            optimizer.step()  # update
            if (i + 1) % 100 == 0:
                # 对训练集（下方2行）和测试集（调用eval）进行评测  100个batch_size进行一次输出
                predict_labels = torch.max(predict, dim=1)[1].numpy()
                true = labels.data
                train_acc = metrics.accuracy_score(true, predict_labels)
                dev_acc, dev_loss = evaluate(config, dev_loader, model, criterion)
                if dev_acc > max_acc:  # 存储最佳acc模型并存储acc图像
                    max_acc = dev_acc
                    best_epoch = epoch
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,'dev_acc': dev_acc, 'dev_acc_ls': dev_acc_ls}
                    torch.save(state, config.model_save_name)
                    improve = '*'  # 在有提升的结果后面加上*标注
                else:
                    improve = ''
                msg = 'Iter: {0:>6}, Train loss: {1:>5.2},' \
                      ' Train acc: {2:>6.2%}, Val loss: {3:>5.2}, Val acc: {4:>6.2%}, Time: {5} {6}'
                print(msg.format(i + 1, loss.item(), train_acc, dev_loss, dev_acc, time_since(start), improve))

        dev_acc, dev_loss = evaluate(config, dev_loader, model, criterion)
        dev_acc_ls.append(dev_acc)
        epoch_np = np.arange(1, epoch + 1, 1)  # 绘制accuracy图像
        acc_np = np.array(dev_acc_ls)
        plt.plot(epoch_np, acc_np)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.savefig('result/Accuracy.png')
        print('EPOCH: {}/{}, dev loss: {:.4f}, dev acc: {:.2f}%, Time usage: {}'.format(
            epoch, config.num_epoch, dev_loss, dev_acc * 100, time_since(start)))
        scheduler.step()  # lr衰减
    print('Best model at epoch {}, del acc: {:.2f}%, '.format(best_epoch, max_acc * 100))
    test(config, model, test_loader, criterion)  # 全程只进行一次测试
    print('Time usage:', time_since(start))

def evaluate(config,data_loader, model, criterion, test=False):
    model.eval()  # 关闭dropout和BatchNormal
    labels_all = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            x, labels = data
            predict = model(x)  # forward
            loss = criterion(predict, labels)
            loss_total += loss
            predict_labels = torch.max(predict, dim=1)[1].numpy()  # ???
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict_labels)
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all,
                                               target_names=config.class_ls, digits=4)  # 包含标签索引  浮点值位数
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        msg = 'Test loss: {0:5>2}, Test acc: {1:6.2%}'
        print(msg.format((loss_total/len(data_loader), acc)))
        print('Precision, Recall and F1-Score:')
        print(report)
        print('Confusion Matrix:')
        print(confusion)
    else:
        return acc, loss_total/len(data_loader)  # 为何这里要除以len(data_loader) 而训练集不除？


def test(config, model, test_loader, criterion):
    ckpt = torch.load(config.model_save_name)
    model.load_state_dict(ckpt['model'])
    evaluate(config, test_loader, model, criterion, test=True)

def create_dir_not_exists(filename):
    if not os.path.exists(filename):
        os.mkdir(filename)

def time_since(since):
    """
    函数说明： 返回已用时间

    Parameter：
    ----------
        since
    Return:
    -------
        time_use - 自since开始所用时间  eg: 00:05:12
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2021-2-1 09:17:58
    """
    time_dif = time.time() - since
    time_use = timedelta(seconds=int(round(time_dif)))
    return time_use
