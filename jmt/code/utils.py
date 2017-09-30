import os

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

def loadEmbeddings(embedding, voc, fileName):
    assert os.path.exists(fileName)

    print('Loading embeddings from '+fileName)

    with open(fileName, 'r') as f:
        counter = 0

        for line in f:
            fields = line.split()

            if len(fields)-1 != embedding.weight.size(1):
                continue

            tokenIndex = voc.getTokenIndex(fields[0])
            
            if tokenIndex == voc.tokenIndex[voc.UNK]:
                continue

            counter += 1

            for i in range(len(fields)-1):
                embedding.weight[tokenIndex][i].data.fill_(float(fields[i+1]))

    print(str(counter)+' embeddings are initialized')
    print()

def getCharNgram(token):
    BEG = '#BEGIN#'
    END = '#END#'
    result = []

    chars = [BEG] + list(token) + [END]
    
    for n in [2, 3, 4]:
        for i in range(len(chars)-n+1):
            result.append(str(n) + 'gram-' + ''.join(chars[i:i+n]))

    return result
