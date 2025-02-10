import numpy as np

def createOutputAsCentroids(data, label):
    centroidLabels = np.unique(label)
    outputData = np.zeros([np.shape(data)[0], np.shape(data)[1]])
    
    for i in range(len(centroidLabels)):
        indices = np.where(label == centroidLabels[i])[0]
        tmpData = data[indices, :]
        centroid = np.mean(tmpData, axis=0)
        outputData[indices, ] = centroid
    
    return outputData

def SLCE(X, labels):
    """
    Implementation of Supervised Linear Centroid Encoder (SLCE).
    
    Eigen decomposition is used to solve the optimization problem.
    
    Parameters:
    X      : [n x d] array where n is the number of samples and d is the number of features.
    labels : [n x 1] array or list with n elements representing sample labels.
    
    Returns:
    eVals  : Eigenvalues in descending order.
    eVecs  : Corresponding eigenvectors.
    """
    # Mean center the data
    X = X - np.mean(X, axis=0)
    
    # Calculate the C matrix
    C = createOutputAsCentroids(X, labels)
    
    # Transpose the matrices X, C to make each column a data point
    X, C = X.T, C.T
    
    # Build the Q matrix
    Q = 2 * np.dot(C, X.T) - np.dot(X, X.T)
    
    # Compute eigenvalues and eigenvectors
    eVals, eVecs = np.linalg.eigh(Q)
    
    # Flip the order to get descending eigenvalues
    eVals, eVecs = np.flip(eVals), np.flip(eVecs, axis=1)
    
    return eVals, eVecs
