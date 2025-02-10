import numpy as np
def runscatter(features, label):
    features = features.T
    label = label.T
    dim = features.shape[0]
    unq =  np.unique(label)
    withinScatter = np.zeros((dim, dim))
    betweenScatter = np.zeros((dim, dim))
    label = label.flatten()
    totalMean = features.mean(axis = 1, keepdims = True)
    for cls in unq:
        lab = label == cls
        clsSamples = features[:, lab]
        classCentroid = clsSamples.mean(axis=1,keepdims = True)
        classCardinality = clsSamples.shape[1]
        # print(classCardinality)
        cMINUSx = clsSamples - classCentroid
        cMINUSxTranspose = cMINUSx.T
        mul = np.matmul(cMINUSx, cMINUSxTranspose)
        mul = mul / classCardinality
        withinScatter = withinScatter + mul
        classCentroidDiff = classCentroid - totalMean
        classCentroidDiffTranspose = classCentroidDiff.T
        mul2 = np.matmul(classCentroidDiff, classCentroidDiffTranspose)
        betweenScatter = betweenScatter + (mul2)

    withinScatter = withinScatter + 0.00001*np.eye(dim)
    wsnorm = np.linalg.norm(withinScatter)
    bsnorm = np.linalg.norm(betweenScatter)
    # print("withinScatter", wsnorm)
    # print("betweenScatter", bsnorm)
    return len(unq), withinScatter, betweenScatter

def embeddMatrix(features, label):
    clsLen, withinScatter, betweenScatter = runscatter(features, label)
    # print("shape", betweenScatter.shape)
    mul = np.matmul(np.linalg.inv(withinScatter),betweenScatter)
    # print(mul.shape)
    mul = np.matmul(np.linalg.inv(withinScatter),betweenScatter)
    eVals,eVecs = np.linalg.eigh(mul)
    # eVals_flip = np.flip(eVals)
    # eVecs_flip = np.flip(eVecs,axis=1)
    
    return clsLen, eVals, eVecs
    

def transform(features, label):
    clsLen, eVals, eVecs = embeddMatrix(features, label)
    eVals_flip = np.flip(eVals)
    eVecs_flip = np.flip(eVecs,axis=1)
    eVals = eVals_flip[:clsLen]
    eVecs = eVecs_flip[:, :clsLen]
    newFeatures = np.matmul(features, eVecs)
    return newFeatures

