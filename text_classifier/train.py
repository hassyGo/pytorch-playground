import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import TextClassifier
from data import Corpus
import utils
import random

embedDim = 50
hiddenDim = embedDim
biDirectional = True

batchSize = 8
initialLearningRate = 1.0
lrDecay = 0.0
gradClip = 1.0
weightDecay = 1.0e-06
maxEpoch = 100

minFreq = 1

useGpu = True

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

classifier = TextClassifier(corpus.voc.size(),
                            embedDim, hiddenDim,
                            corpus.classVoc.size(),
                            biDirectional,
                            repType = 'Sen',
                            actType = 'Tanh')

if useGpu:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        classifier.cuda()
        print('**** Running with GPU ****\n')
    else:
        useGpu = False
        print('**** Warning: GPU is not available ****\n')


criterionClassifier = nn.CrossEntropyLoss(size_average = True)

epoch = 0

batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

while epoch < maxEpoch:
    aveLoss = 0.0
    trainAcc = 0.0

    epoch += 1
    print('--- Epoch '+str(epoch))

    random.shuffle(corpus.trainData)

    opt = optim.SGD(classifier.parameters(),
                    lr = initialLearningRate/(1.0+lrDecay*epoch),
                    weight_decay = weightDecay)
    classifier.train()
    
    for batch in batchListTrain:
        opt.zero_grad()

        # build input for the batch
        curBatchSize = batch[1]-batch[0]+1
        batchInput, batchTarget, lengths = utils.buildBatchInOutForClassifier(corpus.voc.getTokenIndex(corpus.voc.PAD), batch, corpus.trainData)
        target = Variable(batchTarget)

        if biDirectional:
            shape = 2, curBatchSize, hiddenDim
        else:
            shape = 1, curBatchSize, hiddenDim
        h0 = c0 = Variable(torch.zeros(*shape), requires_grad = False)

        if useGpu:
            batchInput = batchInput.cuda()
            target = target.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        output = classifier(batchInput, lengths, (h0, c0))

        loss = criterionClassifier(output, target)
        loss.backward()
        nn.utils.clip_grad_norm(classifier.parameters(), gradClip)
        opt.step()

        _, prediction = torch.max(output, 1)
        trainAcc += torch.sum(torch.eq(prediction, target)).data[0]
        aveLoss += loss.data[0]

    print('Train loss: '+str(aveLoss/len(batchListTrain)))
    print('Train acc.: '+str(100.0*trainAcc/len(corpus.trainData)))

    classifier.eval()
    devAcc = 0.0
    for batch in batchListDev:
        # build input for the batch
        curBatchSize = batch[1]-batch[0]+1
        batchInput, batchTarget, lengths = utils.buildBatchInOutForClassifier(corpus.voc.getTokenIndex(corpus.voc.PAD), batch, corpus.devData)
        target = Variable(batchTarget)

        if biDirectional:
            shape = 2, curBatchSize, hiddenDim
        else:
            shape = 1, curBatchSize, hiddenDim
        h0 = c0 = Variable(torch.zeros(*shape), requires_grad = False)

        if useGpu:
            batchInput = batchInput.cuda()
            target = target.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        output = classifier(batchInput, lengths, (h0, c0))
        _, prediction = torch.max(output, 1)
        devAcc += torch.sum(torch.eq(prediction, target)).data[0]

    print('Dev acc.:   '+str(100.0*devAcc/len(corpus.devData)))
