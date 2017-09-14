import torch
import torch.nn as nn
from torch.autograd import Variable
import math

'''
Text classification model
- Input:  text (words, phrases, sentences, or documents)
- Output: class label
'''
class TextClassifier(nn.Module):

    '''
    Initialize the classifier model
    '''
    def __init__(self, vocSize, embedDim, hiddenDim, classNum, biDirectional, repType, actType):
        super(TextClassifier, self).__init__()

        self.dropout = nn.Dropout(p = 0.0)

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
            
        self.hiddenLayer = nn.Linear(classifierDim, hiddenDim)
        assert actType in {'Tanh', 'ReLU'}
        if actType == 'Tanh':
            self.hiddenAct = nn.Tanh()
        elif actType == 'ReLU':
            self.hiddenAct = nn.ReLU()

        self.softmaxLayer = nn.Linear(hiddenDim, classNum)

        self.embedDim = embedDim
        self.hiddenDim = hiddenDim
        self.classifierDim = classifierDim
        self.biDirectional = biDirectional

        self.initWeights()

    '''
    Initialize the model paramters
    '''
    def initWeights(self):
        initScale = math.sqrt(6.0)/math.sqrt(self.hiddenDim+(self.embedDim+self.hiddenDim))
        initScale2 = math.sqrt(6.0)/math.sqrt(self.classifierDim+(self.hiddenDim))
        
        self.embedding.weight.data.uniform_(-initScale, initScale)

        self.encoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0.data.zero_()
        self.encoder.bias_hh_l0.data.zero_()
        self.encoder.bias_hh_l0.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1
        
        self.hiddenLayer.weight.data.uniform_(-initScale2, initScale2)
        self.hiddenLayer.bias.data.zero_()

        self.softmaxLayer.weight.data.zero_()
        self.softmaxLayer.bias.data.zero_()

    '''
    Compute sentence representations
    '''
    def encode(self, batchInput, lengths, hidden0):
        batchInput = torch.t(batchInput)
        input = self.embedding(Variable(batchInput))
        packedInput = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first = True)

        h, (hn, cn) = self.encoder(packedInput, hidden0)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        
        if self.repType == 'Sen':
            if self.biDirectional:
                a = self.hiddenLayer(torch.cat((hn[0], hn[1]), 1))
            else:
                a = self.hiddenLayer(hn.view(hn.size(1), hn.size(2)))
        elif self.repType == 'Ave':
            assert False
        elif self.repType == 'Max':
            assert False

        return self.hiddenAct(a), h

    '''
    Compute class scores
    '''
    def forward(self, batchInput, lengths, hidden0):
        encoded, _ = self.encode(batchInput, lengths, hidden0)
        output = self.softmaxLayer(encoded.view(len(lengths), self.hiddenDim))
        return output

