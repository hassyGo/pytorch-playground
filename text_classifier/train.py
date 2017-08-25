import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import TextClassifier
from data import Corpus
import utils
import random

embedDim = 30
hiddenDim = 30
biDirectional = True

batchSize = 30
initialLearningRate = 1.0
lrDecay = 0.1
gradClip = 1.0
maxEpoch = 100

minFreq = 2

trainFile = './dataset/stanford_sentiment_sample.train'
devFile = './dataset/stanford_sentiment_sample.dev'

seed = 1

torch.manual_seed(seed)
random.seed(seed)

corpus = Corpus(trainFile, devFile, minFreq)

print('Vocabulary size: '+str(corpus.voc.size()))
print('# of classes:    '+str(corpus.classVoc.size()))
print()
print('# of training samples: '+str(len(corpus.trainData)))
print('# of dev samples:      '+str(len(corpus.devData)))

model = TextClassifier(corpus.voc.size(),
                       embedDim, hiddenDim,
                       corpus.classVoc.size(),
                       biDirectional,
                       repType = 'Sen',
                       actType = 'Tanh')
criterion = nn.CrossEntropyLoss(size_average = True)

epoch = 0

batchList = utils.buildBatchList(len(corpus.trainData), batchSize)

while epoch < maxEpoch:
    aveLoss = 0.0
    trainAcc = 0
    opt = optim.SGD(model.parameters(), lr = initialLearningRate/(1.0+lrDecay*epoch))

    epoch += 1
    print('--- Epoch '+str(epoch))

    random.shuffle(corpus.trainData)
    
    for batch in batchList:
        opt.zero_grad()

        # build input for the batch
        curBatchSize = batch[1]-batch[0]+1
        batchInput, batchTarget, lengths = utils.buildBatchInputTarget(corpus.voc.getTokenIndex(corpus.voc.PAD), batch, corpus.trainData)
        target = Variable(batchTarget)

        if biDirectional:
            shape = 2, curBatchSize, hiddenDim
        else:
            shape = 1, curBatchSize, hiddenDim
        h0 = c0 = Variable(torch.zeros(*shape), requires_grad = False)
        output = model(batchInput, lengths, (h0, c0))

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), gradClip)
        opt.step()

        _, prediction = torch.max(output, 1)
        trainAcc += torch.sum(torch.eq(prediction, target)).data[0]
        aveLoss += loss.data[0]

    print('Train loss: '+str(aveLoss/len(batchList)))
    print('Train acc.: '+str(100.0*trainAcc/len(corpus.trainData)))
