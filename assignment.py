import numpy as np
import matplotlib.pyplot as plt
import pickle as p
import random
from mpl_toolkits.mplot3d import Axes3D
from sampleSelectionNplots import sampleSelctionNplots
from SLCE import SLCE
from LDA import embeddMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
import time



# Load dataset
data_path = "assignment1 datasets/Digits.p"
labels_path = "assignment1 datasets/Labels.p"

with open(data_path, 'rb') as file:
    features = p.load(file) 

with open(labels_path, 'rb') as file:
    labels = p.load(file)  
    

X = features
L = labels
# eps = 1e-8
random.seed(17)
uniqueClss = np.unique(L)

def eigenProjection(eVecs):
    # Project data into 2D and 3D space
    projectedData2D = np.matmul(X, eVecs[:, :2])
    projectedData3D = np.matmul(X, eVecs[:, :3])
    return projectedData2D, projectedData3D

def pca(X, L):
    mu = np.mean(X, axis=0)
    X = X - mu
   
    eVals, eVecs = np.linalg.eigh(np.dot(X.T, X) / (X.shape[0] - 1))
    eVals_flip, eVecs_flip = np.flip(eVals), np.flip(eVecs, axis = 1)
    return eVals_flip, eVecs_flip

def pcaImplementation(X, L):
    
    eVals_flip, eVecs_flip = pca(X, L)
    projectedData2D, projectedData3D = eigenProjection(eVecs_flip)
    samplesNeeded = 100
    sampleSelctionNplots(projectedData2D, projectedData3D, uniqueClss, samplesNeeded, L, "PCA")
    totalVariance = np.sum(eVals_flip)
    eVals_flip = eVals_flip/totalVariance
    cumsumeVal = np.cumsum(eVals_flip)*100
    vecs = np.argmax(cumsumeVal >= 99) + 1
    print("Total eigenvectors are needed to capture the 99% of the total variance of the data is ",  vecs)
    plt.plot(range(len(cumsumeVal)), cumsumeVal, label="Cumulative Variance")
    plt.axvline(x=vecs, color='red', linestyle='--', label=f'99% Variance')
    plt.text(vecs + 10, 99 - 5, f"99% at {vecs} eigenvectors", color='red')
    plt.xlabel(f"Number of eigen vectors")
    plt.ylabel(f"cumulative sum percentage")
    plt.title(f"cumulative sum percentage vs Number of eigen vectors")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    # plt.legend(loc="upper right", title="cumulative sum")
    plt.savefig(f"cumulative sum vs Number of eigen vectors.jpeg")
    plt.clf()
    eigenVecList = list(range(0, vecs+10, 10))
    colors = cm.tab10.colors
    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(uniqueClss):
        indices = np.where(L == cls)[0]
        clsList = X[indices, :]
        n = clsList.shape[0]
        errorVal = []
        for eigenVector in eigenVecList:
            eigenVector = eigenVector + 1
            eVecsTotalVariance = eVecs_flip[:, :eigenVector]
            newX = np.matmul(clsList, eVecsTotalVariance)
            reconstructedX = np.dot(newX, eVecsTotalVariance.T)
            fNorm = reconstructedX - clsList
            reconstructionError = np.linalg.norm(fNorm) / n
            errorVal.append(reconstructionError)
            
        name = "Digit Class " + str(cls)
        plt.plot(eigenVecList, errorVal, color = colors[i % 10], label=name, marker='o')
    plt.xlabel(f"Number of eigen vectors")
    plt.ylabel(f"Forbenious error")
    plt.title(f"Reconstruction error vs Eigen vectors of digit classes")
    plt.legend(loc="upper right", title="Digit Classes")  
    plt.grid(True)
    plt.savefig(f"Reconstruction error of digit classes.jpeg")
    # plt.show()
    plt.clf()   

    return eVals_flip, eVecs_flip 


def accCal(eVecsPCA, embeddDim, itr):
    knn = KNeighborsClassifier(n_neighbors=5)
    avgAcc = []
    for p in embeddDim:
        accList = []
        eVecsTotalVariance = eVecsPCA[:, :p]
        newX = np.matmul(X, eVecsTotalVariance)
        for i in range(itr):
            Xtrain, Xtest, Ltrain, Ltest = train_test_split(newX, L, test_size=0.2, stratify=L, random_state=None)
            Ltrain = Ltrain.ravel()
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(Xtrain, Ltrain)
            LPred = knn.predict(Xtest)
            acc = accuracy_score(Ltest, LPred)
            accList.append(acc)
        pACC = np.mean(accList)
        avgAcc.append(pACC)
    avgAcc = np.array(avgAcc)*100
    return avgAcc


def knnACC(X, L):
    eValsPCA, eVecsPCA = pca(X, L)
    clsLenLDA, eValsLDA, eVecsLDA = embeddMatrix(X, L)
    eValsLDA, eVecsLDA = np.flip(eValsLDA), np.flip(eVecsLDA, axis = 1)
    eValsSLCE, eVecsSLCE = SLCE(X, L)
    embeddDim = list(range(1, 10, 2))
    avgAccPCA = accCal(eVecsPCA, embeddDim, 10)
    avgAccLDA = accCal(eVecsLDA, embeddDim, 10)
    avgAccSLCE = accCal(eVecsSLCE, embeddDim, 10)
    # colors = cm.tab10.colors
    plt.figure()
    plt.plot(embeddDim, avgAccPCA, color='b', marker='o', label = "PCA")
    plt.plot(embeddDim, avgAccLDA, color='g', marker='o', label = "LDA")
    plt.plot(embeddDim, avgAccSLCE, color='r', marker='o', label = "SLCE")
    plt.xlabel(f"Embedding dimension")
    plt.ylabel(f"5nn Avg Accuracy")
    plt.title(f"Accuracy vs Embedding dimension for various DR's")
    plt.legend(loc="upper left", title="DR") 
    plt.grid(True)
    plt.savefig(f"Accuracy vs Embedding dimension.jpeg")
    # plt.show()
    plt.clf()

def pcaReconstruction(X, L):
    eVals, eVecs = pca(X, L)
    embeddDim = list(range(11, 161, 10))
    errorVal = []
    for eigenVector in embeddDim:
        eigenVector = eigenVector + 1
        eVecsTotalVariance = eVecs[:, :eigenVector]
        newX = np.matmul(X, eVecsTotalVariance)
        reconstructedX = np.dot(newX, eVecsTotalVariance.T)
        fNorm = reconstructedX - X
        reconstructionError = np.linalg.norm(fNorm) / X.shape[0]
        errorVal.append(reconstructionError)

    errorVal = np.array(errorVal)*100
    avgAccPCA = accCal(eVecs, embeddDim, 10)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(embeddDim, avgAccPCA, color='r', marker='+', label = "PCA")
    ax[0].set_xlabel(f"Embedding dimension")
    ax[0].set_ylabel(f"5nn Avg Accuracy")
    ax[0].set_title(f"Accuracy vs Embedding dimensio of PCA")
    ax[0].legend(loc="upper right", title="PCA 5NN Accuracy") 
    ax[0].grid(True)
    # ax[0].savefig(f"PCA 5NN Accuracy vs Embedding dimension.jpeg")
    ax[1].plot(embeddDim, errorVal, color='y', marker='+', label = "PCA")
    ax[1].set_xlabel(f"Embedding dimension")
    ax[1].set_ylabel(f"Reconstruction error")
    ax[1].set_title(f"Reconstruction error vs Embedding dimension of PCA")
    ax[1].legend(loc="upper right", title="PCA Reconstruction error") 
    ax[1].grid(True)


    fig.savefig("Reconstruction error of PCA.jpeg")
    plt.tight_layout()
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    startTime = time.time()
    pcaImplementation(X, L)
    knnACC(X, L)
    pcaReconstruction(X, L)
    endTime = time.time()
    timeTaken = (endTime - startTime)
    print(f"Time taken: {timeTaken:.4f} seconds")