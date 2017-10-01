import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random
import math
import argparse
import itertools
import os

from data import Corpus
import utils
from model import Embedding
from model import Tagger

parser = argparse.ArgumentParser(description = 'A Joint Many-Task Model')
parser.add_argument('--embedDim', type = int, default = 100,
                    help='Size of word embeddings')
parser.add_argument('--charDim', type = int, default = 100,
                    help='Size of char embeddings')
parser.add_argument('--hiddenDim', type = int, default = 100,
                    help='Size of hidden layers')
parser.add_argument('--batchSize', type = int, default = 32,
                    help='Mini-batch size')
parser.add_argument('--lr', type = float, default = 1.0,
                    help='Initial learning rate')
parser.add_argument('--lrDecay', type = float, default = 0.3,
                    help='Learning rate decay per epoch')
parser.add_argument('--lstmWeightDecay', type = float, default = 1.0e-06,
                    help='Weight decay for LSTM weights')
parser.add_argument('--mlpWeightDecay', type = float, default = 1.0e-05,
                    help='Weight decay for MLP weights')
parser.add_argument('--epoch', type = int, default = 20,
                    help='Maximum number of training epochs')
parser.add_argument('--seed', type = int, default = 1,
                    help='Random seed')
parser.add_argument('--gpuId', type = int, default = 0,
                    help='GPU id')
parser.add_argument('--inputDropout', type = float, default = 0.2,
                    help='Dropout rate for input vectors')
parser.add_argument('--outputDropout', type = float, default = 0.2,
                    help='Dropout rate for output vectors')
parser.add_argument('--clip', type = float, default = 1.0,
                    help='Gradient clipping value')
parser.add_argument('--random', action = 'store_true',
                    help='Use randomly initialized embeddings or not')
parser.add_argument('--test', action = 'store_true',
                    help = 'Test mode or not')

args = parser.parse_args()
print(args)
print()

embedDim = args.embedDim
charDim = args.charDim
hiddenDim = args.hiddenDim
batchSize = args.batchSize
initialLearningRate = args.lr
lrDecay = args.lrDecay
lstmWeightDecay = args.lstmWeightDecay
mlpWeightDecay = args.mlpWeightDecay
maxEpoch = args.epoch
seed = args.seed
inputDropoutRate = args.inputDropout
outputDropoutRate = args.outputDropout
gradClip = args.clip
useGpu = True
gpuId = args.gpuId
test = args.test

trainFile = '../dataset/pos/pos_wsj.sample.train'
devFile = '../dataset/pos/pos_wsj.sample.dev'

wordEmbeddingFile = '../embedding/word.txt'
charEmbeddingFile = '../embedding/charNgram.txt'

modelParamsFile = 'params-'+str(gpuId)
embeddingParamsFile = 'embedding-'+str(gpuId)
wordParamsFile = 'word_params-'+str(gpuId) # for pre-trained embeddings
charParamsFile = 'char_params-'+str(gpuId) # for pre-trained embeddings

torch.manual_seed(seed)
random.seed(seed)

corpus = Corpus(trainFile, devFile)

print('Vocabulary size: '+str(corpus.voc.size()))
print('# of classes:    '+str(corpus.classVoc.size()))
print()
print('# of training samples: '+str(len(corpus.trainData)))
print('# of dev samples:      '+str(len(corpus.devData)))
print()

embedding = Embedding(corpus.voc.size(), corpus.charVoc.size(), embedDim, charDim)
tagger = Tagger(embedDim+charDim, hiddenDim, corpus.classVoc.size(),
                inputDropoutRate, outputDropoutRate)

if not test and not args.random:
    if os.path.exists(wordParamsFile):
        embedding.wordEmbedding.load_state_dict(torch.load(wordParamsFile))
    else:
        utils.loadEmbeddings(embedding.wordEmbedding, corpus.voc, wordEmbeddingFile)
        torch.save(embedding.wordEmbedding.state_dict(), wordParamsFile)

    if os.path.exists(charParamsFile):
        embedding.charEmbedding.load_state_dict(torch.load(charParamsFile))
    else:
        utils.loadEmbeddings(embedding.charEmbedding, corpus.charVoc, charEmbeddingFile)
        torch.save(embedding.charEmbedding.state_dict(), charParamsFile)

if useGpu:
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuId)
        torch.cuda.manual_seed(seed)
        embedding.cuda()
        tagger.cuda()
        print('**** Running with GPU-' + str(args.gpuId) + ' ****\n')
    else:
        useGpu = False
        print('**** Warning: GPU is not available ****\n')

criterionTagger = nn.CrossEntropyLoss(size_average = False, ignore_index = -1)

batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

totalParams = list(embedding.parameters())+list(tagger.parameters())
lstmParams = []
mlpParams = []
withoutWeightDecay = []
for name, param in list(embedding.named_parameters())+list(tagger.named_parameters()):
    if not param.requires_grad:
        continue
    if 'bias' in name or 'Embedding' in name:
        withoutWeightDecay += [param]
    elif 'encoder' in name:
        lstmParams += [param]
    else:
        mlpParams += [param]
optParams = [{'params': lstmParams, 'weight_decay': lstmWeightDecay},
             {'params': mlpParams, 'weight_decay': mlpWeightDecay},
             {'params': withoutWeightDecay, 'weight_decay': 0.0}]

opt = optim.SGD(optParams,
                lr = initialLearningRate)

maxDevAcc = -100.0
epoch = 0

while epoch < maxEpoch and not test:
    trainAcc = 0.0
    trainTokenCount = 0.0
    batchProcessed = 0

    for paramGroup in opt.param_groups:
        paramGroup['lr'] = initialLearningRate/(1.0+lrDecay*epoch)

    epoch += 1
    print('--- Epoch '+str(epoch))

    random.shuffle(corpus.trainData)
    embedding.train()
    tagger.train()
    
    '''
    Mini-batch training
    '''
    for batch in batchListTrain:
        opt.zero_grad()
        batchInput, batchChar, batchTarget, lengths, hidden0, tokenCount = corpus.processBatchInfo(batch, True, hiddenDim, useGpu)
        trainTokenCount += tokenCount

        output = tagger(embedding.getBatchedEmbedding(batchInput, batchChar), lengths, hidden0)
        loss = criterionTagger(output, batchTarget)
        loss /= (batch[1]-batch[0]+1.0)
        loss.backward()
        nn.utils.clip_grad_norm(totalParams, gradClip)
        opt.step()

        _, prediction = torch.max(output, 1)
        trainAcc += (prediction.data == batchTarget.data).sum()

        batchProcessed += 1
        '''
        Mini-batch test
        '''
        if batchProcessed == len(batchListTrain)//20:
            batchProcessed = 0
            devAcc = 0.0
            devTokenCount = 0.0

            embedding.eval()
            tagger.eval()
            for batch in batchListDev:
                batchInput, batchChar, batchTarget, lengths, hidden0, tokenCount = corpus.processBatchInfo(batch, False, hiddenDim, useGpu)
                devTokenCount += tokenCount
        
                output = tagger(embedding.getBatchedEmbedding(batchInput, batchChar), lengths, hidden0)
                _, prediction = torch.max(output, 1)
                devAcc += (prediction.data == batchTarget.data).sum()
            embedding.train()
            tagger.train()

            devAcc = 100.0*devAcc/devTokenCount
            print('Dev acc.:   '+str(devAcc))

            if devAcc > maxDevAcc:
                maxDevAcc = devAcc
                torch.save(tagger.state_dict(), modelParamsFile)
                torch.save(embedding.state_dict(), embeddingParamsFile)

    print('Train acc.: '+str(100.0*trainAcc/trainTokenCount))

if test:
    tagger.load_state_dict(torch.load(modelParamsFile))
    embedding.load_state_dict(torch.load(embeddingParamsFile))

    embedding.eval()
    tagger.eval()

    devAcc = 0.0
    devTokenCount = 0.0
    for batch in batchListDev:
        batchInput, batchChar, batchTarget, lengths, hidden0, tokenCount = corpus.processBatchInfo(batch, False, hiddenDim, useGpu)
        devTokenCount += tokenCount

        output = tagger(embedding.getBatchedEmbedding(batchInput, batchChar), lengths, hidden0)
        _, prediction = torch.max(output, 1)
        devAcc += (prediction.data == batchTarget.data).sum()

    devAcc = 100.0*devAcc/devTokenCount
    print('Dev acc.:   '+str(devAcc))

    embedding.train()
    tagger.train()
