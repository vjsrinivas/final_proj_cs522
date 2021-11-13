import numpy as np
from matplotlib import pyplot as plt
from pca import pca

def standardizeData(datatr, datate):
    datatrT = datatr.T
    
    means = []
    for d in datatrT:
        avg = sum(d)/len(d)
        means.append(avg)

    sigmas = []
    for i, mean in enumerate(means):
        variance = sum([((x - mean) ** 2) for x in datatrT[i]]) / len(datatrT[i])
        res = variance ** 0.5
        sigmas.append(res)

    for i in range(len(datatr)):
        for j in range(len(datatr[i])-1):
            datatr[i][j] = (datatr[i][j] - means[j])/sigmas[j]
    
    for i in range(len(datate)):
        for j in range(len(datate[i])-1):
            datate[i][j] = (datate[i][j] - means[j])/sigmas[j]
    return datatr, datate

datatr = np.loadtxt("data/pima.tr.txt")
datate = np.loadtxt("data/pima.te.txt")

correctClass = datate[:,-1].astype(int)

#standardize of the pima dataset
datatr, datate = standardizeData(datatr, datate)

xtrain = datatr[:, :-1]
ytrain = datatr[:, -1].astype(int)

xtest = datate[:, :-1]
ytest = datate[:, -1].astype(int)

print("Pima dataset with original 7 dimenstions")
print()
print(xtrain)
print()
redxtrain = pca(xtrain, 2)
print()
print("Pima dataset with dimensions reduced to 2")
print()
print(redxtrain)
print()