from data import Corpus
from model import Embedding
from model import EncDec
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
import os
import time
import sys

# En-Ja dev
sourceDevFile = './data/sample.en.dev'
sourceOrigDevFile = './data/sample.en.dev'
targetDevFile = './data/sample.ja.dev'

# En-Ja train
sourceTrainFile = './data/sample.en'
sourceOrigTrainFile = './data/sample.en'
targetTrainFile = './data/sample.ja'

minFreqSource = 2 # use source-side words which appear at least N times in the training data
minFreqTarget = 2 # use target-side words which appear at least N times in the training data
hiddenDim = 128   # dimensionality of hidden states and embeddings
decay = 0.5       # learning rate decay rate for SGD
gradClip = 1.0    # clipping value for gradient-norm clipping
dropoutRate = 0.2 # dropout rate for output MLP
numLayers = 1     # number of LSTM layers (1 or 2)
    
maxLen = 100      # use sentence pairs whose maximum lengths are 100 in both source and target sides
maxEpoch = 20
decayStart = 5

sourceEmbedDim = hiddenDim
targetEmbedDim = hiddenDim

batchSize = 16    # "128" is typically used
learningRate = 1.0
momentumRate = 0.75

gpuId = [0, 1, 2, 3]
seed = int(sys.argv[1])

device = torch.device('cuda:'+str(gpuId[0]))
cpu = torch.device('cpu')

weightDecay = 1.0e-06

train = True

beamSize = 10

if not train:
    batchSize = 1

torch.set_num_threads(1)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

corpus = Corpus(sourceTrainFile, sourceOrigTrainFile, targetTrainFile, sourceDevFile, sourceOrigDevFile, targetDevFile, minFreqSource, minFreqTarget, maxLen)
    
print('Source vocabulary size: '+str(corpus.sourceVoc.size()))
print('Target vocabulary size: '+str(corpus.targetVoc.size()))
print()
print('# of training samples: '+str(len(corpus.trainData)))
print('# of develop samples:  '+str(len(corpus.devData)))
print('SEED: ', str(seed))
print()

embedding = Embedding(sourceEmbedDim, targetEmbedDim, corpus.sourceVoc.size(), corpus.targetVoc.size())
encdec = EncDec(sourceEmbedDim, targetEmbedDim, hiddenDim, corpus.targetVoc.size(), dropoutRate = dropoutRate, numLayers = numLayers)

encdec.wordPredictor.softmaxLayer.weight = embedding.targetEmbedding.weight
encdec.wordPredictor = nn.DataParallel(encdec.wordPredictor, gpuId)

if train:
    embedding.to(device)
    encdec.to(device)

batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

withoutWeightDecay = []
withWeightDecay = []
for name, param in list(embedding.named_parameters())+list(encdec.named_parameters()):
    if 'bias' in name or 'Embedding' in name:
        withoutWeightDecay += [param]
    elif 'softmax' not in name:
        withWeightDecay += [param]
optParams = [{'params': withWeightDecay, 'weight_decay': weightDecay},
             {'params': withoutWeightDecay, 'weight_decay': 0.0}]
totalParamsNMT = withoutWeightDecay+withWeightDecay

opt = optim.SGD(optParams, momentum = momentumRate, lr = learningRate)

bestDevGleu = -1.0
prevDevGleu = -1.0

for epoch in range(maxEpoch):
    if not train:
        break

    batchProcessed = 0
    totalLoss = 0.0
    totalTrainTokenCount = 0.0
    
    print('--- Epoch ' + str(epoch+1))
    startTime = time.time()
    
    random.shuffle(corpus.trainData)

    embedding.train()
    encdec.train()

    for batch in batchListTrain:
        print('\r', end = '')
        print(batchProcessed+1, '/', len(batchListTrain), end = '')
        
        batchSize = batch[1]-batch[0]+1

        opt.zero_grad()

        batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData = corpus.processBatchInfoNMT(batch, train = True, device = device)
        
        inputSource = embedding.getBatchedSourceEmbedding(batchInputSource)
        sourceH, (hn, cn) = encdec.encode(inputSource, lengthsSource)
        
        batchInputTarget = batchInputTarget.to(device)
        batchTarget = batchTarget.to(device)
        inputTarget = embedding.getBatchedTargetEmbedding(batchInputTarget)
        
        loss = encdec(inputTarget, lengthsTarget, lengthsSource, (hn, cn), sourceH, batchTarget)
        loss = loss.sum()

        totalLoss += loss.item()
        totalTrainTokenCount += tokenCount

        loss /= batchSize
        loss.backward()
        nn.utils.clip_grad_norm_(totalParamsNMT, gradClip)
        opt.step()

        batchProcessed += 1
        if batchProcessed == len(batchListTrain)//2 or batchProcessed == len(batchListTrain):
            devPerp = 0.0
            devGleu = 0.0
            totalTokenCount = 0.0

            embedding.eval()
            encdec.eval()
            torch.set_grad_enabled(False)

            print()
            print('Training time: ' + str(time.time()-startTime) + ' sec')
            print('Train perp: ' + str(math.exp(totalLoss/totalTrainTokenCount)))
            
            f_trans = open('./trans.txt', 'w')
            f_gold = open('./gold.txt', 'w')

            for batch in batchListDev:
                batchSize = batch[1]-batch[0]+1
                batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData = corpus.processBatchInfoNMT(batch, train = False, device = device)

                inputSource = embedding.getBatchedSourceEmbedding(batchInputSource)
                sourceH, (hn, cn) = encdec.encode(inputSource, lengthsSource)

                indicesGreedy, lengthsGreedy, attentionIndices = encdec.greedyTrans(corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, maxGenLen = maxLen)
                indicesGreedy = indicesGreedy.to(cpu)

                for i in range(batchSize):
                    for k in range(lengthsGreedy[i]-1):
                        index = indicesGreedy[i, k].item()
                        if index == corpus.targetVoc.unkIndex:
                            index = attentionIndices[i, k].item()
                            f_trans.write(batchData[i].sourceOrigStr[index] + ' ')
                        else:
                            f_trans.write(corpus.targetVoc.tokenList[index].str + ' ')
                    f_trans.write('\n')

                    for k in range(lengthsTarget[i]-1):
                        index = batchInputTarget[i, k+1].item()
                        if index == corpus.targetVoc.unkIndex:
                            f_gold.write(batchData[i].targetUnkMap[k] + ' ')
                        else:
                            f_gold.write(corpus.targetVoc.tokenList[index].str + ' ')
                    f_gold.write('\n')

                batchInputTarget = batchInputTarget.to(device)
                batchTarget = batchTarget.to(device)
                inputTarget = embedding.getBatchedTargetEmbedding(batchInputTarget)
                
                loss = encdec(inputTarget, lengthsTarget, lengthsSource, (hn, cn), sourceH, batchTarget)
                loss = loss.sum()
                devPerp += loss.item()

                totalTokenCount += tokenCount

            f_trans.close()
            f_gold.close()
            os.system("./bleu.sh 2> DUMMY")
            f_trans = open('./bleu.txt', 'r')
            for line in f_trans:
                devGleu = float(line.split()[2][0:-1])
                break
            f_trans.close()
            
            devPerp = math.exp(devPerp/totalTokenCount)
            print("Dev perp:", devPerp)
            print("Dev BLEU:", devGleu)
            
            embedding.train()
            encdec.train()
            torch.set_grad_enabled(True)

            if epoch > decayStart and devGleu < prevDevGleu:
                print('lr -> ' + str(learningRate*decay))
                learningRate *= decay

                for paramGroup in opt.param_groups:
                    paramGroup['lr'] = learningRate

            elif devGleu >= bestDevGleu:
                bestDevGleu = devGleu

                stateDict = embedding.state_dict()
                for elem in stateDict:
                    stateDict[elem] = stateDict[elem].to(cpu)
                torch.save(stateDict, './params/embedding.bin')

                stateDict = encdec.state_dict()
                for elem in stateDict:
                    stateDict[elem] = stateDict[elem].to(cpu)
                torch.save(stateDict, './params/encdec.bin')
                
            prevDevGleu = devGleu

if train:
    exit(0)

torch.set_grad_enabled(False)

embedding.load_state_dict(torch.load('./params/embedding.bin'))
encdec.load_state_dict(torch.load('./params/encdec.bin'))

embedding.to(device)
encdec.to(device)

embedding.eval()
encdec.eval()

f_trans = open('./trans.txt', 'w')
f_gold = open('./gold.txt', 'w')

devPerp = 0.0
totalTokenCount = 0.0

for batch in batchListDev:
    batchSize = batch[1]-batch[0]+1
    batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData = corpus.processBatchInfoNMT(batch, train = False, device = device)

    inputSource = embedding.getBatchedSourceEmbedding(batchInputSource)
    sourceH, (hn, cn) = encdec.encode(inputSource, lengthsSource)

    if beamSize == 1:
        indicesGreedy, lengthsGreedy, attentionIndices = encdec.greedyTrans(corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, maxGenLen = maxLen)
    else:
        indicesGreedy, lengthsGreedy, attentionIndices = encdec.beamSearch(corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, beamSize = beamSize, maxGenLen = maxLen)
    indicesGreedy = indicesGreedy.to(cpu)

    for i in range(batchSize):
        for k in range(lengthsGreedy[i]-1):
            index = indicesGreedy[i, k].item()
            if index == corpus.targetVoc.unkIndex:
                index = attentionIndices[i, k].item()
                f_trans.write(batchData[i].sourceOrigStr[index] + ' ')
            else:
                f_trans.write(corpus.targetVoc.tokenList[index].str + ' ')
        f_trans.write('\n')

        for k in range(lengthsTarget[i]-1):
            index = batchInputTarget[i, k+1].item()
            if index == corpus.targetVoc.unkIndex:
                f_gold.write(batchData[i].targetUnkMap[k] + ' ')
            else:
                f_gold.write(corpus.targetVoc.tokenList[index].str + ' ')
        f_gold.write('\n')

    batchInputTarget = batchInputTarget.to(device)
    batchTarget = batchTarget.to(device)
    inputTarget = embedding.getBatchedTargetEmbedding(batchInputTarget)

    loss = encdec(inputTarget, lengthsTarget, lengthsSource, (hn, cn), sourceH, batchTarget)
    loss = loss.sum()
    devPerp += loss.item()

    totalTokenCount += tokenCount

f_trans.close()
f_gold.close()
os.system("./bleu.sh 2> DUMMY")
f_trans = open('./bleu.txt', 'r')
for line in f_trans:
    devGleu = float(line.split()[2][0:-1])
    break
f_trans.close()

devPerp = math.exp(devPerp/totalTokenCount)
print("Dev perp:", devPerp)
print("Dev BLEU:", devGleu)
