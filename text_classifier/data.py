import os
import torch

class Token:
    def __init__(self, str_ = '', count_ = 0):
        self.str = str_
        self.count = count_

class Vocabulary:
    def __init__(self):
        self.UNK = '<UNK>' # unkown words
        self.EOS = '<EOS>' # the end-of-sequence token
        self.BOS = '<BOS>' # the beginning-of-sequence token
        self.PAD = '<PAD>' # padding
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
    def __init__(self, text_, label_):
        self.text = text_
        self.label = label_

class Corpus:
    def __init__(self, trainFile = '', devFile = '', minFreq = 2):
        self.voc = Vocabulary()
        self.classVoc = Vocabulary()

        self.buildVoc(trainFile, minFreq)
        self.trainData = self.buildDataset(trainFile)
        self.devData = self.buildDataset(devFile)

    def buildVoc(self, fileName, minFreq):
        assert os.path.exists(fileName)

        with open(fileName, 'r') as f:
            tokenCount = {}
            unkCount = 0
            eosCount = 0

            labelCount = {}

            for line in f:
                tokens = line.split('\t')[1].split() # label \t w1 w2 ... \n
                label = line.split('\t')[0]
                eosCount += 1
                
                for t in tokens:
                    if t in tokenCount:
                        tokenCount[t] += 1
                    else:
                        tokenCount[t] = 1

                if label in labelCount:
                    labelCount[label] += 1
                else:
                    labelCount[label] = 1

            # select words which appear >= minFreq
            tokenList = sorted(tokenCount.items(), key = lambda x: -x[1]) # sort by value
            labelList = sorted(labelCount.items(), key = lambda x: -x[1]) # sort by value
            
            for t in tokenList:
                if t[1] >= minFreq:
                    self.voc.add(t[0], t[1])
                else:
                    unkCount += t[1]
            self.voc.add(self.voc.UNK, unkCount)
            self.voc.add(self.voc.BOS, eosCount)
            self.voc.add(self.voc.EOS, eosCount)
            self.voc.add(self.voc.PAD, 0)

            for l in labelList:
                self.classVoc.add(l[0], l[1])

    def buildDataset(self, fileName):
        assert os.path.exists(fileName)

        with open(fileName, 'r') as f:
            dataset = []
            
            for line in f:
                tokens = [self.voc.BOS] + line.split('\t')[1].split() + [self.voc.EOS] # label \t w1 w2 ... \n
                tokenIndices = torch.LongTensor(len(tokens))
                label = torch.LongTensor(1)
                i = 0

                for t in tokens:
                    tokenIndices[i] = self.voc.getTokenIndex(t)
                    i += 1
                
                label[0] = self.classVoc.getTokenIndex(line.split('\t')[0])
                dataset.append(Data(tokenIndices, label))

        return dataset
