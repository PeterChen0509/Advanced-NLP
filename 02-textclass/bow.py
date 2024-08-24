from collections import defaultdict
import time
import random
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from model import BoW

# Function to read corpus
# 每当字典 w2i 中出现一个新的键（即之前未出现的键），这个函数就会被调用，并返回当前 w2i 的长度
w2i = defaultdict(lambda: len(w2i)) # 将单词和标签映射到一个唯一的整数ID
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        # 函数读取一个文件，并将每一行分成两个部分：标签（tag）和词（words），然后将这些词转换为整数ID，并将结果作为一个生成器返回
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("/home/peterchen/Study/Advanced-NLP/data/sst-sentiment-text-fiveclass/train.txt"))
# w2i 的默认行为从之前的 lambda: len(w2i)（自动分配唯一ID）变成了 lambda: UNK（返回一个固定的默认值）
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("/home/peterchen/Study/Advanced-NLP/data/sst-sentiment-text-fiveclass/test.txt"))
nwords = len(w2i) # 词汇表的大小
ntags = len(t2i) # 标签的数量

# initialize model
model = BoW(nwords, ntags)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        words = torch.tensor(words).type(type)
        tag = torch.tensor([tag]).type(type)
        scores = model(words)
        loss = criterion(scores, tag)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" %(
        ITER, train_loss/len(train), time.time()-start
    ))
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
        words = torch.tensor(words).type(type)
        scores = model(words)[0].detach().cpu().numpy()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" %(ITER, test_correct/len(dev)))