import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model import WordPredictor

import math

class MultiHeadAttention(nn.Module):

    def __init__(self, embedDim, headNum, dropoutRate):
        super(MultiHeadAttention, self).__init__()

        assert embedDim%headNum == 0

        self.dropout = nn.Dropout(p = dropoutRate)

        self.wK = Parameter(torch.FloatTensor(headNum, embedDim, embedDim//headNum))
        self.wQ = Parameter(torch.FloatTensor(headNum, embedDim, embedDim//headNum))
        self.wV = Parameter(torch.FloatTensor(headNum, embedDim, embedDim//headNum))
        self.wO = nn.Linear(embedDim, embedDim, bias = False)

        self.layernorm = nn.LayerNorm(embedDim)

        self.embedDim = embedDim
        self.headNum = headNum

        self.initWeight()

    def initWeight(self):
        scale = 1.0/math.sqrt(self.embedDim) #0.1

        self.wK.data.uniform_(-scale, scale)
        self.wQ.data.uniform_(-scale, scale)
        self.wV.data.uniform_(-scale, scale)
        self.wO.weight.data.uniform_(-scale, scale)

    '''
    kv:   (B, Lkv, D)
    q:    (B, Lq,  D))
    mask: (B*H, Lq, Lkv)

    output: (B, Lq, D)
    '''
    def forward(self, kv, q, mask):
        batchSize = kv.size(0)
        Lkv = kv.size(1)
        Lq = q.size(1)

        K = torch.matmul(kv.unsqueeze(1), self.wK) # (B, H, Lkv, d)
        V = torch.matmul(kv.unsqueeze(1), self.wV) # (B, H, Lkv, d)
        Q = torch.matmul(q.unsqueeze(1), self.wQ)  # (B, H, Lq, d)

        K = K.view(K.size(0)*K.size(1), K.size(2), K.size(3))
        V = V.view(V.size(0)*V.size(1), V.size(2), V.size(3))
        Q = Q.view(Q.size(0)*Q.size(1), Q.size(2), Q.size(3))

        QK = torch.bmm(Q, K.transpose(1, 2)) # (B*H, Lq, Lkv)
        QK /= math.sqrt(self.embedDim/self.headNum)
        QK = QK + mask
        QK = F.softmax(QK, dim = 2)
        
        multiHead = torch.bmm(QK, V) # (B*H, Lq, d)
        multiHead = multiHead.transpose(1, 2).contiguous().view(batchSize, self.embedDim, Lq).transpose(1, 2) # (B, Lq, D)
        multiHead = self.wO(multiHead)

        q = self.dropout(q)
        multiHead = self.layernorm(multiHead+q)

        return multiHead

class FF(nn.Module):

    def __init__(self, embedDim, hiddenDim, dropoutRate):
        super(FF, self).__init__()

        self.dropout = nn.Dropout(p = dropoutRate)

        self.ff1 = nn.Linear(embedDim, hiddenDim)
        self.ff2 = nn.Linear(hiddenDim, embedDim)

        self.layernorm = nn.LayerNorm(embedDim)

        #self.initWeight()

    def initWeight(self):
        scale = 0.1

        self.ff1.weight.data.uniform_(-scale, scale)
        self.ff1.bias.data.zero_()

        self.ff2.weight.data.uniform_(-scale, scale)
        self.ff2.bias.data.zero_()

    def forward(self, input):
        output = self.ff1(input)
        output = torch.relu(output)
        input = self.dropout(input)
        output = self.layernorm(self.ff2(output) + input)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embedDim, hiddenDim, headNum, dropoutRate):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(embedDim, headNum, dropoutRate)
        self.ff = FF(embedDim, hiddenDim, dropoutRate)

    def forward(self, input, mask):
        output = self.attn(input, input, mask)
        output = self.ff(output)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, embedDim, hiddenDim, headNum, dropoutRate):
        super(TransformerDecoderLayer, self).__init__()

        self.attn1 = MultiHeadAttention(embedDim, headNum, dropoutRate)
        self.attn2 = MultiHeadAttention(embedDim, headNum, dropoutRate)
        self.ff = FF(embedDim, hiddenDim, dropoutRate)
        
    def forward(self, input, kv, mask_encoder, mask_decoder):
        output = self.attn1(input, input, mask_decoder)
        output = self.attn2(kv, output, mask_encoder)
        output = self.ff(output)

        return output

class Transformer(nn.Module):
    
    def __init__(self, embedDim, hiddenDim, headNum, layerNum, targetVocSize, dropoutRate):
        super(Transformer, self).__init__()

        self.dropout = nn.Dropout(p = dropoutRate)

        tmp = [TransformerEncoderLayer(embedDim, hiddenDim, headNum, dropoutRate) for i in range(layerNum)]
        self.encoder = nn.Sequential(*tmp)
        tmp = [TransformerDecoderLayer(embedDim, hiddenDim, headNum, dropoutRate) for i in range(layerNum)]
        self.decoder = nn.Sequential(*tmp)

        self.wordPredictor = WordPredictor(embedDim, targetVocSize)
        
        self.embedDim = embedDim
        self.layerNum = layerNum

        maxLen = 1024
        freq = 10000
        self.pe = torch.FloatTensor(maxLen, self.embedDim)
        for j in range(maxLen):
            for i in range(self.embedDim):
                if i%2 == 0:
                    self.pe.data[j, i] = math.sin((j+1)/math.pow(freq, 2*(i+1)/self.embedDim))
                else:
                    self.pe.data[j, i] = math.cos((j+1)/math.pow(freq, 2*(i+1)/self.embedDim))

    def forward(self, sourceInput, sourceMask, targetInput, targetMask, label = None):
        sourceH = math.sqrt(self.embedDim) * sourceInput + self.pe[:sourceInput.size(1), :].unsqueeze(0)
        targetH = math.sqrt(self.embedDim) * targetInput + self.pe[:targetInput.size(1), :].unsqueeze(0)

        sourceH = self.dropout(sourceH)
        targetH = self.dropout(targetH)

        for i in range(self.layerNum):
            sourceH = self.encoder[i](sourceH, sourceMask[:, :sourceH.size(1), :])
            targetH = self.decoder[i](targetH, sourceH, sourceMask[:, :targetH.size(1), :], targetMask)
            
        targetH = targetH.contiguous().view(targetH.size(0)*targetH.size(1), targetH.size(2))
        output = self.wordPredictor(targetH, label)

        return output

