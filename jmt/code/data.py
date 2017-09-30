import os
import torch

from torch.autograd import Variable

import utils

class Token:
    def __init__(self, str_ = '', count_ = 0):
        self.str = str_
        self.count = count_

class Vocabulary:
    def __init__(self):
        self.UNK = '<UNK>' # unkown words
        self.PAD = '<PAD>' # padding
        self.unkIndex = -1
        self.padIndex = -1
        self.tokenIndex = {}
        self.tokenList = []

    def getTokenIndex(self, str):
        if str in self.tokenIndex:
            return self.tokenIndex[str]
        else:
            return self.tokenIndex[self.UNK]

    def add(self, str, count):
        if str not in self.tokenIndex:
            self.tokenList.append(Token(str, count))
            self.tokenIndex[str] = len(self.tokenList)-1

    def size(self):
        return len(self.tokenList)

class Data:
    def __init__(self, tokenIndices_, charNgramIndices_, labelIndices_):
        self.tokenIndices = tokenIndices_
        self.charNgramIndices = charNgramIndices_
        self.labelIndices = labelIndices_

class Corpus:
    def __init__(self, trainFile = '', devFile = ''):
        self.voc = Vocabulary()
        self.classVoc = Vocabulary()
        self.charVoc = Vocabulary()

        self.buildVoc(trainFile)
        self.trainData = self.buildDataset(trainFile)
        self.devData = self.buildDataset(devFile)

    def buildVoc(self, fileName):
        assert os.path.exists(fileName)

        with open(fileName, 'r') as f:
            tokenCount = {}
            charNgramCount = {}
            labelCount = {}

            for line in f:
                fields = line.split('\t')
                tokens = fields[0].split() # w1 w2 ... wn \t l1 l2 ... ln \n
                labels = fields[1].split()
                assert len(tokens) == len(labels)
                
                for t in tokens:
                    for c in utils.getCharNgram(t):
                        if c in charNgramCount:
                            charNgramCount[c] += 1
                        else:
                            charNgramCount[c] = 1

                    t = t.lower()
                    if t in tokenCount:
                        tokenCount[t] += 1
                    else:
                        tokenCount[t] = 1

                for l in labels:
                    if l in labelCount:
                        labelCount[l] += 1
                    else:
                        labelCount[l] = 1

            tokenList = sorted(tokenCount.items(), key = lambda x: -x[1]) # sort by value
            charNgramList = sorted(charNgramCount.items(), key = lambda x: -x[1]) # sort by value
            labelList = sorted(labelCount.items(), key = lambda x: -x[1]) # sort by value

            for t in tokenList:
                self.voc.add(t[0], t[1])
            for c in charNgramList:
                self.charVoc.add(c[0], c[1])
            for l in labelList:
                self.classVoc.add(l[0], l[1])

            '''
            Add special tokens
            '''
            self.voc.add(self.voc.UNK, 0)
            self.voc.add(self.voc.PAD, 0)
            self.voc.unkIndex = self.voc.getTokenIndex(self.voc.UNK)
            self.voc.padIndex = self.voc.getTokenIndex(self.voc.PAD)
            self.charVoc.add(self.charVoc.UNK, 0)
            self.charVoc.add(self.charVoc.PAD, 0) # use this for padding
            self.charVoc.unkIndex = self.charVoc.getTokenIndex(self.charVoc.UNK)
            self.charVoc.padIndex = self.charVoc.getTokenIndex(self.charVoc.PAD)

            '''
            Prob for UNK word-dropout
            '''
            alpha = 0.25
            for t in self.voc.tokenList:
                t.count = alpha/(t.count + alpha)

    def buildDataset(self, fileName):
        assert os.path.exists(fileName)

        with open(fileName, 'r') as f:
            dataset = []
            
            for line in f:
                fields = line.split('\t')
                tokens = fields[0].split() # w1 w2 ... wn \t l1 l2 ... ln \n
                labels = fields[1].split() # w1 w2 ... wn \t l1 l2 ... ln \n
                assert len(tokens) == len(labels)
                tokenIndices = []
                charNgramIndices = []
                labelIndices = []

                for i in range(len(tokens)):
                    charNgramIndices.append([])
                    for c in utils.getCharNgram(tokens[i]):
                        ci = self.charVoc.getTokenIndex(c)
                        if ci != self.charVoc.unkIndex:
                            charNgramIndices[i].append(ci)
                    if len(charNgramIndices[i]) == 0:
                        charNgramIndices[i].append(self.charVoc.unkIndex)

                    tokenIndices.append(self.voc.getTokenIndex(tokens[i].lower()))
                    labelIndices.append(self.classVoc.getTokenIndex(labels[i]))

                dataset.append(Data(tokenIndices, charNgramIndices, labelIndices))

        return dataset

    '''
    input:  w1, w2, ..., wn
    target: l1, l2, ..., ln
    '''
    def processBatchInfo(self, batch, train, hiddenDim, useGpu):
        begin = batch[0]
        end = batch[1]
        batchSize = end-begin+1
        if train:
            data = sorted(self.trainData[begin:end+1], key = lambda x: -len(x.tokenIndices))
        else:
            data = sorted(self.devData[begin:end+1], key = lambda x: -len(x.tokenIndices))
        maxLen = len(data[0].tokenIndices)
        batchInput = torch.LongTensor(batchSize, maxLen).fill_(self.voc.padIndex)
        batchTarget = torch.LongTensor(batchSize*maxLen).fill_(-1)
        lengths = []
        targetIndex = 0
        tokenCount = 0
        
        for i in range(batchSize):
            l = len(data[i].tokenIndices)
            lengths.append(l)
            tokenCount += l

            for j in range(l):
                batchInput[i][j] = data[i].tokenIndices[j]

            for j in range(maxLen):
                if j < l:
                    batchTarget[targetIndex] = data[i].labelIndices[j]
                targetIndex += 1

            '''
            UNK word-dropout
            '''
            if train:
                rnd = torch.FloatTensor(l).uniform_(0.0, 1.0)
                for j in range(l):
                    if rnd[j] < self.voc.tokenList[batchInput[i][j]].count:
                        batchInput[i][j] = self.voc.unkIndex
        assert(targetIndex == batchSize*maxLen)

        batchInput = Variable(batchInput, requires_grad = False)
        batchTarget = Variable(batchTarget, requires_grad = False)

        '''
        Char n-gram
        '''
        batchCharInput = []
        batchCharOffset = []
        offsetPos = 0
        for i in range(batchSize):
            for j in range(maxLen):
                batchCharOffset.append(offsetPos)
                if j < lengths[i]:
                    index = data[i].tokenIndices[j]
                    offsetPos += len(data[i].charNgramIndices[j])
                    batchCharInput += data[i].charNgramIndices[j]
                else:
                    offsetPos += 1
                    batchCharInput.append(self.charVoc.padIndex)

        batchCharInput = Variable(torch.LongTensor(batchCharInput))
        batchCharOffset = Variable(torch.LongTensor(batchCharOffset))

        shape = 2, batchSize, hiddenDim
        h0 = c0 = Variable(torch.zeros(*shape), requires_grad = False)

        if useGpu:
            batchInput = batchInput.cuda()
            batchCharInput = batchCharInput.cuda()
            batchCharOffset = batchCharOffset.cuda()
            batchTarget = batchTarget.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()

        return batchInput, (batchCharInput, batchCharOffset), batchTarget, lengths, (h0, c0), tokenCount
