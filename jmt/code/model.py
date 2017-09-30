import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

'''
Sequential tagging model
- Input:  w1 w2 ... wn
- Output: l1 l2 ... ln
'''
class Tagger(nn.Module):

    '''
    Initialize the tagging model
    '''
    def __init__(self, vocSize, charVocSize, embedDim, charDim, hiddenDim, classNum, inputDropoutRate, outputDropoutRate):
        super(Tagger, self).__init__()

        self.embedding = nn.Embedding(vocSize, embedDim)
        self.charEmbedding = nn.EmbeddingBag(charVocSize, charDim)

        self.encoder = nn.LSTM(input_size = embedDim+charDim,
                               hidden_size = hiddenDim,
                               num_layers = 1,
                               dropout = 0.0,
                               bidirectional = True)
        
        self.inputDropout = nn.Dropout(p = inputDropoutRate)
        self.outputDropout = nn.Dropout(p = outputDropoutRate)
        
        classifierDim = 2*hiddenDim
        self.hiddenLayer = nn.Linear(classifierDim, classifierDim)
        self.hiddenAct = nn.ReLU()

        self.softmaxLayer = nn.Linear(classifierDim, classNum)

        self.embedDim = embedDim
        self.charDim = charDim
        self.hiddenDim = hiddenDim
        self.classifierDim = classifierDim

        self.initWeights()

    '''
    Initialize the model paramters
    '''
    def initWeights(self):
        initScale = math.sqrt(6.0)/math.sqrt(self.hiddenDim+(self.embedDim+self.charDim+self.hiddenDim))
        initScale2 = math.sqrt(6.0)/math.sqrt(self.classifierDim+(self.classifierDim))
        
        self.embedding.weight.data.uniform_(-initScale, initScale)
        self.charEmbedding.weight.data.uniform_(-initScale, initScale)

        self.encoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0.data.zero_()
        self.encoder.bias_hh_l0.data.zero_()
        self.encoder.bias_hh_l0.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

        self.encoder.weight_ih_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0_reverse.data.zero_()
        self.encoder.bias_hh_l0_reverse.data.zero_()
        self.encoder.bias_hh_l0_reverse.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1
        
        self.hiddenLayer.weight.data.uniform_(-initScale2, initScale2)
        self.hiddenLayer.bias.data.zero_()

        self.softmaxLayer.weight.data.zero_()
        self.softmaxLayer.bias.data.zero_()

    '''
    Get mini-batched word embedding matrix
    '''
    def getBatchedEmbedding(self, batchInput, batchChar):
        wordInput = self.embedding(batchInput)

        charInput = self.charEmbedding(batchChar[0], batchChar[1])
        charInput = charInput.view(wordInput.size(0), wordInput.size(1), charInput.size(1))

        return self.inputDropout(torch.cat((wordInput, charInput), dim = 2))

    '''
    Compute feature Vectors
    '''
    def encode(self, input, lengths, hidden0):
        packedInput = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first = True)

        h, (hn, cn) = self.encoder(packedInput, hidden0)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        h = h.contiguous().view(h.size(0)*h.size(1), h.size(2))
        h = self.hiddenLayer(self.outputDropout(h))
        return self.hiddenAct(h)

    '''
    Compute class scores
    '''
    def forward(self, input, lengths, hidden0):
        encoded = self.encode(input, lengths, hidden0)
        return self.softmaxLayer(self.outputDropout(encoded))
