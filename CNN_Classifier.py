# coding: UTF-8
# 测试环境pytorch0.4 + python 3.6，还有其他一些包比如numpy，sklearn这些机器学习常用的包
# 数据集是CNews的长文本分类数据，包含50000个训练样本，5000个验证样本，10000个测试样本，
# 中文的新闻语料，这些新闻共分为体育、时尚、游戏等10个类别。因为是长文本，所以文本预处理的长度阈值设置为了256个token。
# 将数据分为train.txt, dev.txt, test.txt三个文件，这三个文件每一行都是 [原始文本+ '\t' + 类别索引] 的格式。
# 接着再准备一个class.txt文件，每一行是一个类别名，注意第一行对应的类别索引=0，第二行对应的类别索引=1，依此类推，
# 之所以需要这个class.txt是为了后续的测试结果可视化。
# 模型的输入采用预训练的词向量。因为是做中文数据集，所以采用了维基百科中文预训练的词向量 sgns.wiki.word 。
# 该词向量可以到网上下载。在训练时，需要把train.txt, dev.txt, test.txt, class.txt
# 这四个文件以及词向量sgns.wiki.word文件放到以下文件目录：（注意，py文件与CNews文件夹在同一个目录下）

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

from sklearn import metrics
from importlib import import_module
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import timedelta

import pkuseg  # 分词工具包，如果没有，可以用jieba分词，或者用其他分词工具


#################################################################################
############################## 定义各种参数、路径 #################################

class Config(object):
    def __init__(self, dataset_dir, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset_dir + '/data/train.txt'  # 训练集
        self.dev_path = dataset_dir + '/data/dev.txt'  # 验证集
        self.test_path = dataset_dir + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset_dir + '/data/class.txt').readlines()]  # 类别名单
        self.vocab_path = dataset_dir + '/data/vocab.pkl'  # 词表
        self.save_path = dataset_dir + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset_dir + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset_dir + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.4
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20
        self.batch_size = 32
        self.pad_size = 256  # 每句话处理成的长度，截长、补短
        self.learning_rate = 1e-5
        self.embed_dim = self.embedding_pretrained.size(1)  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


#################################################################################
################################### 定义模型结构 #################################

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)
        # 三个卷积层分别是(1, channels=256, kernal_size=(2, 300))
        #                (1, 256, (3, 300))    (1, 256, (4, 300))
        # 这三个卷积层是并行的，同时提取2-gram、3-gram、4-gram特征
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    # 假设embed_dim=300，每个卷积层的卷积核都有256个（会将一个输入seq映射到256个channel上）
    # 三个卷积层分别为：(1, 256, (2, 300)), (1, 256, (3, 300)), (1, 256, (4, 300))
    # x(b_size, 1, seq_len, 300)进入卷积层后得到 (b, 256, seq_len-1, 1), (b, 256, seq_len-2, 1), (b, 256, seq_len-3, 1)
    # 卷积之后经过一个relu，然后把最后一个维度上的1去掉(squeeze)，得到x(b, 256, seq_len-1), 接着进入池化层
    # 一个池化层输出一个(b, 256),三个池化层输出三个(b, 256), 然后在forward里面把三个结果concat起来
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # max_pool1d表示一维池化，一维的意思是，输入x的维度除了b_size和channel，只有一维，即x(b_size, channel, d1)，
        # 故池化层只需要定义一个宽度表示kernel_size
        # max_pool2d表示二维池化，x(b_size, channel, d1, d2), 所以max_pool2d定义的kernel_size是二维的
        # max_pool1d((b, 256, seq_len-1), kernel_size = seq_len-1) -> (b, 256, 1)
        # squeeze(2) 之后得到 (b, 256)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    """
    nn中的成员比如nn.Conv2d，都是类，可以提取待学习的参数。当我们在定义网络层的时候，层内如果有需要学习的参数，那么我们就要用nn组件；
    nn.functional里的成员都是函数，只是完成一些功能，比如池化，整流线性函数，不保存参数，所以如果某一层只是单纯完成一些简单的功能，没有
    待学习的参数，那么就用nn.funcional里的组件
    """

    # 后续数据预处理时候，x被处理成是一个tuple,其形状是: (data, length).
    # 其中data(b_size, seq_len),  length(batch_size)
    # x[0]:(b_size, seq_len)
    def forward(self, x):
        out = self.embedding(x[0])  # x[0]:(b_size, seq_len, embed_dim)   x[1]是一维的tensor,表示batch_size个元素的长度
        out = out.unsqueeze(1)  # (b_size, 1, seq_len, embed_dim)  在第1维增加一个维度（注意是以0开始）与squeeze方法相反
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # (b, channel * 3) == (b, 256 * 3)
        out = self.dropout(out)
        out = self.fc(out)  # out(b, num_classes)
        return out


# 泽维尔正态分布 xavier_normal_：均值为0，标准差为根号(2/(输入+输出数))的正态分布，默认gain=1
# kaiming正态分布 kaiming_normal_：均值为0，标准差为根号(2/(1+a²)f_in)的正态分布，默认a=0
# 初始化时候要避开预训练词向量
def init_network(model, method='xavier', exclude='embedding', seed=123):  # 没看懂   ？？？
    for name, w in model.named_parameters():
        if exclude not in name:  # 对于embedding，保留预训练的embedding
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


#################################################################################
################################### 训练、测试过程 ################################

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()  # 存在dropout及BatchNormal时使用  model.eval也一样
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每个epoch：学习率 = gamma * 学习率
    # 配合 scheduler.step() 完成学习率衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')  # 'inf' 正无穷 '-inf' 负无穷
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # from tensorboardX import SummaryWriter  记录训练的日志
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()  # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):  # 每个(train_iter)相当于 -> ((x[b, len], len[b]), labels[b])
            outputs = model(trains)  # trains[0]:(b_size, seq_len)保存idx的二维tensor,  trains[1]:(b_size)表示长度的一维tensor
            # 1.清空梯度 -> 2.计算loss -> 3.反向传播 -> 4.梯度更新
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)  # outputs(b, num_classes), labels(b)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)  # sklearn.metrics.accuracy_score(true, predic) 返回正确的比例
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
                      '  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()  # 因为调用evaluate时，evaluate会调用model.eval()，从而使得dropout失效
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()  # 关闭dropout
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():  # 将outputs从计算图中排除
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # cpu()表示把值放到cpu  numpy()表示把值转化为np类型
            # append拼接数组和数值，也可以拼接两个数组，数组必须是np.array()类型，拼接成一维的np.array()
            labels_all = np.append(labels_all, labels)   # 保存所有正确的输入标签
            predict_all = np.append(predict_all, predic)    # 保存所有预测标签

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # classification_report用于显示每个class上的各项指标结果，包括precision, recall, f1-score
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # 混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)  # 混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


#######################################################################################
################################## 数据预处理过程 ######################################

seg = pkuseg.pkuseg()  # 分词工具，通过 seg.cut(x) 进行分词。 可以换成 jieba 分词

MAX_VOC_SIZE = 500000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()  # 去除字符串头尾的空格或换行符
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 对每个词计数 并返回对应字典  eg: 我：10
        # vocab_list = sorted([_ for _ in vocab_dic.items() \
        # if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)  # ???  下面几行不理解
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        # tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        tokenizer = lambda x: seg.cut(x)  # 分词
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOC_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0, len), ([...], 1, len), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    """__init__
        batches: [([...], 0, len), ([...], 1, len), ...],  每个元素是(seq_idx_list[], label, len)，是全部样本(未分批)
        batch_size: config.batch_size
        device: config.device
    """

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # False表示没有余数，代表n_batch数量是整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    """input
        datas是一个batch_size的样本，格式相当于batches，每个元素都是(seq_idx_list[], label, len)
    output
        (x[b_size, max_len], len[b_size]), label[b_size]
        x是padding后的idx张量，
        len是这个batch里每个样本padding之前的长度，超过pad_size的长度定为pad_size
        label是每个样本真实的标签
    """

    def _to_tensor(self, datas):
        # x:(b_size, seq_len) -> [[...], [...], ...] 一个batch_size，代表seq
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # y:(b_size) -> [0, 1, 5, ...] 一个batch_size，代表标签
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)   seq_len:(b_size) -> [len1, len2, ...]
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y  # (x[b, l], l[b]), y[b]

    # 当被作为迭代器访问时(比如for循环时)，__next__方法给出了一定的规则来依次返回类的成员
    def __next__(self):
        # 假设len(self.batches)==25, self.n_batches==4, self.batch_size==6
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]  # 最后余下的不足batch_size数量的样本
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif not self.residue and self.index == self.n_batches:  # 这一句待定
            self.index = 0
            raise StopIteration
        elif self.index > self.n_batches:  # 指示batch的索引归零，当前epoch结束
            self.index = 0
            raise StopIteration  # 迭代停止的异常
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):  # 使DatasetIterater的实例变成可迭代对象。
        # __iter__需要返回一个迭代器。由于类里同时定义了__next__方法，__iter__返回实例本身就可以
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    # timedelta返回时间间隔
    return timedelta(seconds=int(round(time_dif)))


# 下面的目录、文件名按需更改。
train_dir = "./CNews/data/train.txt"
vocab_dir = "./CNews/data/vocab.pkl"
pretrain_dir = "./CNews/data/sgns.wiki.word"
emb_dim = 300
filename_trimmed_dir = "./CNews/data/myEmbedding"
if os.path.exists(vocab_dir):
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
else:
    # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    tokenizer = lambda x: seg.cut(x)  # 以词为单位构建词表
    word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOC_SIZE, min_freq=1)
    pkl.dump(word_to_id, open(vocab_dir, 'wb'))

embeddings = np.random.rand(len(word_to_id), emb_dim)
f = open(pretrain_dir, "r", encoding='UTF-8')
for i, line in enumerate(f.readlines()):
    # if i == 0:  # 若第一行是标题，则跳过
    #     continue
    lin = line.strip().split(" ")
    if lin[0] in word_to_id:
        idx = word_to_id[lin[0]]
        emb = [float(x) for x in lin[1:301]]
        embeddings[idx] = np.asarray(emb, dtype='float32')
f.close()
np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

#######################################################################################
################################## 加载数据，开始训练 ###################################

# dataset文件夹名称
dataset_dir = 'CNews'
# 词向量名称
embedding_file = 'myEmbedding.npz'

config = Config(dataset_dir, embedding_file)
# 中文的话使用字符级别的词向量，字符串里每个字符就是一个字
use_word = True  # True for word, False for char

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()

print("Loading data...")
# train -> # [([...], 0), ([...], 1), ...]
vocab, train_data, dev_data, test_data = build_dataset(config, use_word)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
config.n_vocab = len(vocab)  # 词表大小

model_name = config.model_name
model = Model(config).to(config.device)
init_network(model)  # 初始化隐层参数
print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter)