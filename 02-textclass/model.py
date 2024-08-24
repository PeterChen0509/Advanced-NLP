""" 
BoW 模型：最基础的模型，直接对嵌入向量求和后输出分类结果。
CBoW 模型：在 BoW 的基础上增加了线性层，使得模型能够捕捉到更多的模式。
DeepCBoW 模型：进一步增加了多层线性层和非线性激活函数，使得模型更具表达力，适合更复杂的任务
"""
import torch
from torch import nn
from torch.autograd import Variable

class BoW(torch.nn.Module):
    def __init__(self, nwords, ntags): # 词汇表中的词汇数, 类别标签的数量
        super(BoW, self).__init__()
        # variables
        type = torch.FloatTensor
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            type = torch.cuda.FloatTensor
        
        self.bias = Variable(torch.zeros(ntags), requires_grad=True).type(type)
        # layers
        self.embedding = nn.Embedding(nwords, ntags)
        # 初始化权重xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, words):
        emb = self.embedding(words)
        out = torch.sum(emb, dim=0) + self.bias
        out = out.view(1,-1)
        return out

class CBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, emb_size): # 词汇表中的词汇数, 类别标签的数量, 嵌入向量的维度
        super(CBoW, self).__init__()
        # layers
        self.embedding = nn.Embedding(nwords, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linear = nn.Linear(emb_size, ntags) # bias 是默认带的
        nn.init.xavier_uniform_(self.linear.weight)
    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0) # size(emb_sum) = emb_size
        emb_sum = emb_sum.view(1,-1) # size(emb_sum) = 1 x emb_size
        out = self.linear(emb_sum) # size(out) = 1 x ntags
        return out

class DeepCBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, nlayers, emb_size, hid_size): # 词汇表中的词汇数, 类别标签的数量, 网络中隐藏层的数量, 嵌入向量的维度, 隐藏层的维度
        super(DeepCBoW, self).__init__()
        # variables
        self.nlayers = nlayers
        # layers
        self.embedding = nn.Embedding(nwords, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # add nlayers number of layers
        self.linears = nn.ModuleList([
            nn.Linear(emb_size if i == 0 else hid_ize, hid_size) for i in range(nlayers)
        ])
        # 初始化
        for i in range(nlayers):
            nn.init.xavier_uniform_(self.linears[i].weight)
        
        self.output_layer = nn.Linear(hid_size, ntags)
        nn.init.xavier_uniform_(self.output_layer.weight)
    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0) # size(emb_sum) = emb_size
        h = emb_sum.view(1,-1) # size(h) = 1 x emb_size
        for i in range(self.nlayers):
            h = torch.tanh(self.linears[i](h)) # 经过多层线性变换，并在每层之后应用 tanh 激活函数
        out = self.output_layer(h)
        return out