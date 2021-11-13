## PCA code 
## Program will reduce the dimensions of a given data set to the specifed dimensions using principal component analysis
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def pca(data, d):
    pcaObject = PCA(n_components=d)
    reducedData = pcaObject.fit_transform(data)
    return reducedData

def main():
    print("Performing PCA on Data")


if __name__ == "__main__":
    main()