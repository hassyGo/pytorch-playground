import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class TextClassifier(nn.Module):

    def __init__(self, vocSize, embedDim, hiddenDim, classNum, biDirectional, repType, actType):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocSize, embedDim)

        self.encoder = nn.LSTM(input_size = embedDim,
                               hidden_size = hiddenDim,
                               num_layers = 1,
                               dropout = 0.0,
                               bidirectional = biDirectional)

        classifierDim = hiddenDim
        if biDirectional:
            classifierDim *= 2

        assert repType in {'Sen', 'Ave', 'Max'}
        self.repType = repType
            
        self.hiddenLayer = nn.Linear(classifierDim, classifierDim)
        assert actType in {'Tanh', 'ReLU'}
        if actType == 'Tanh':
            self.hiddenAct = nn.Tanh()
        elif actType == 'ReLU':
            self.hiddenAct = nn.ReLU()

        self.softmaxLayer = nn.Linear(classifierDim, classNum)

        self.embedDim = embedDim
        self.hiddenDim = hiddenDim
        self.classifierDim = classifierDim
        self.biDirectional = biDirectional

        self.initWeights()
        
    def initWeights(self):
        initScale = math.sqrt(6.0)/math.sqrt(self.hiddenDim+(self.embedDim+self.hiddenDim))
        initScale2 = math.sqrt(6.0)/math.sqrt(self.classifierDim+(self.classifierDim))
        
        self.embedding.weight.data.uniform_(-initScale, initScale)

        self.encoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0.data.zero_()
        self.encoder.bias_hh_l0.data.zero_()
        self.encoder.bias_hh_l0.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1
        
        self.hiddenLayer.weight.data.uniform_(-initScale2, initScale2)
        self.hiddenLayer.bias.data.fill_(0.0)

        self.softmaxLayer.weight.data.fill_(0.0)
        self.softmaxLayer.bias.data.fill_(0.0)

    def forward(self, batchInput, lengths, hidden):
        input = self.embedding(Variable(batchInput))
        packedInput = nn.utils.rnn.pack_padded_sequence(input, lengths)

        h, (hn, cn) = self.encoder(packedInput, hidden)

        if self.repType == 'Sen':
            if self.biDirectional:
                a = self.hiddenLayer(torch.cat((hn[0], hn[1]), 1))
            else:
                a = self.hiddenLayer(hn.view(hn.size(1), hn.size(2)))
            hidden = self.hiddenAct(a)
            output = self.softmaxLayer(hidden.view(len(lengths), self.classifierDim))
        elif self.repType == 'Ave':
            assert False
        elif self.repType == 'Max':
            assert False
        return output
