import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

def sampleSelctionNplots(projectedData2D, projectedData3D, uniqueClss, samplesNeeded, L, name):
    smallSamples = []
    for digit in uniqueClss:
        indices = np.where(L == digit)[0]
        minNum = min(samplesNeeded, len(indices))
        randomClsSample = random.sample(list(indices), minNum)
        smallSamples.extend(randomClsSample)

    projectedData2D = projectedData2D[smallSamples]
    projectedData3D = projectedData3D[smallSamples]
    L = L[smallSamples]

    # 2D Scatter Plot
    plt.figure(figsize=(10, 8))
    distribution2D = plt.scatter(projectedData2D[:, 0], projectedData2D[:, 1], c=L, cmap='tab10', alpha=0.7, marker='o')
    colors = distribution2D.cmap(distribution2D.norm(np.unique(L)))
    handles = [
        plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=str(label))
        for color, label in zip(colors, np.unique(L))
    ]    
    plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1, 1), loc='upper left', frameon=True)
    plt.xlabel(f"{name} Axis 1")
    plt.ylabel(f"{name} Axis 2")
    plt.title(f"{name} Projection in 2D")
    plt.savefig(f"{name}2D.jpeg")
    # plt.show()
    plt.clf()

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    distribution3D = ax.scatter(projectedData3D[:, 0], projectedData3D[:, 1], projectedData3D[:, 2], c=L, cmap='tab10', alpha=0.7)
    colors = distribution3D.cmap(distribution3D.norm(np.unique(L)))
    ax.set_xlabel(f"{name} Axis 1")
    ax.set_ylabel(f"{name} Axis 2")
    ax.set_zlabel(f"{name} Axis 3")
    ax.set_title(f"{name} Projection in 3D")
    handles = [
        plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=str(label))
        for color, label in zip(colors, np.unique(L))
    ]
    ax.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.savefig(f"{name}3D.jpeg")
    # plt.show()
    plt.clf()