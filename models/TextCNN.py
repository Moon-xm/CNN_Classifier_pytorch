# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class config(object):
    def __init__(self):
        # 路径类 带*的是运行前的必要文件  未带*文件/文件夹若不存在则训练过程会生成
        self.train_path = 'data/train.txt'  # *
        self.dev_path = 'data/dev.txt'  # *
        self.class_ls_path = 'data/class.txt'  # *
        self.pretrain_dir = 'data/sgns.sogou.char'  # 前期下载的预训练词向量*
        self.test_path = 'data/test.txt'  # 若该文件不存在会加载dev.txt进行最终测试 不推荐*
        self.vocab_path = 'data/vocab.pkl'
        self.model_save_dir = 'checkpoint'
        self.model_save_name = self.model_save_dir + '/TextCNN.ckpt'  # 保存最佳dev acc模型

        # 可调整的参数
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz,  若不存在则后期生成
        # 随机初始化:random
        self.embedding_type = 'embedding_SougouNews.npz'
        self.use_gpu = True  # 是否使用gpu(有则加载 否则自动使用cpu)
        self.batch_size = 128
        self.pad_size = 32  # 句子长度限制  短补(<PAD>)长截
        self.num_epochs = 40  # 训练轮数
        self.num_workers = 0  # 启用多线程
        self.learning_rate = 0.001  # 训练发现0.001比0.01收敛快(Adam)
        self.embedding_dim = 300  # 词嵌入维度
        self.num_filters = 256  # 卷积核数量（channels数）
        self.require_improvement = 1  # 1个epoch若在dev上acc未提升则自动结束

        # 由前方参数决定  不用修改
        self.class_ls = [x.strip() for x in open(self.class_ls_path, 'r', encoding='utf-8').readlines()]
        self.num_class = len(self.class_ls)
        self.vocab_len = 0  # 词表大小(训练集总的字数(字符级)） 在embedding层作为参数 后期赋值
        self.embedding_pretrained = None  # 根据config.embedding_type后期赋值  random:None  else:tensor from embedding_type
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


class ConvPool(nn.Module):  # conv -> relu -> maxpool
    def __init__(self, config):
        super(ConvPool, self).__init__()
        self.conv1 = nn.Conv2d(1, config.num_filters, kernel_size=(2, config.embedding_dim))  # (B, 1, S, e_d) -> (B, 256, S-1, 1)
        self.conv2 = nn.Conv2d(1, config.num_filters, kernel_size=(3, config.embedding_dim))
        self.conv3 = nn.Conv2d(1, config.num_filters, kernel_size=(4, config.embedding_dim))

    def forward(self, x):
        out1 = self.conv1(x).squeeze(3)  # (B, 1, S, 300) ->(B, 256, S-1, 1) -> (B, 256, S-1)
        out1 = F.relu(out1)
        out1 = F.max_pool1d(out1, out1.size(2))  # (B, 256, S-1) -> (B, 256, 1)
        out2 = self.conv2(x).squeeze(3)
        out2 = F.relu(out2)
        out2 = F.max_pool1d(out2, out2.size(2))
        out3 = self.conv3(x).squeeze(3)
        out3 = F.relu(out3)
        out3 = F.max_pool1d(out3, out3.size(2))

        out = torch.cat([out1, out2, out3], dim=1).squeeze(2)  # 沿通道数拼接 (B, 256*3)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,
                                                          freeze=False)  # 表示训练过程词嵌入向量会更新
        else:
            self.embedding = nn.Embedding(config.vocab_len, config.embedding_dim,
                                          padding_idx=config.vocab_len - 1)  # PAD索引填充 (B, 1, S) -> (B, 1, S, e_d)
        self.conv_pool = ConvPool(config)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(config.num_filters * 3, config.num_class)  # -> (B, num_class)

    def forward(self, x):
        # 数据预处理时，x被处理成是一个tuple,其形状是: (data, length).
        # 其中data(b_size, seq_len),  length(batch_size)
        # x[0]:(b_size, seq_len)
        x = self.embedding(x[0])  # -> (B, S, e_d)
        x = x.unsqueeze(1)  # ->(B, 1, S, e_d)
        x = self.conv_pool(x)  # ->(B, 256*3)
        x = self.drop(x)  # 随机失活
        x = self.fc(x)  # (B, 256*3) -> (B, num_class)
        return x

