import torch

def buildBatchList(dataSize, batchSize):
    batchList = []
    numBatch = int(dataSize/batchSize)

    for i in range(numBatch):
        batch = []
        batch.append(i*batchSize)
        if i == numBatch-1:
            batch.append(dataSize-1)
        else:
            batch.append((i+1)*batchSize-1)
        batchList.append(batch)

    return batchList

def buildBatchInOutForClassifier(paddingIndex, batch, trainData):
    begin = batch[0]
    end = batch[1]
    batchSize = end-begin+1
    data = sorted(trainData[begin:end+1], key = lambda x: -len(x.text))
    maxLen = len(data[0].text)
    batchInput = torch.LongTensor(maxLen, batchSize)
    batchInput.fill_(paddingIndex)
    batchTarget = torch.LongTensor(batchSize)
    lengths = []
    
    for i in range(batchSize):
        batchTarget[i] = data[i].label[0]
        l = len(data[i].text)
        lengths.append(l)
        for j in range(l):
            batchInput[j][i] = data[i].text[j]

    return batchInput, batchTarget, lengths
