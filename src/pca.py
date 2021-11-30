## PCA code 
## Program will reduce the dimensions of a given data set to the specifed dimensions using principal component analysis
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from scipy import sparse

def pca(data, d):
    pcaObject = PCA(n_components=d)
    reducedData = pcaObject.fit_transform(data)
    return reducedData

def incremental_pca(data, d, batch_size):
    pcaObject = IncrementalPCA(n_components=d, batch_size=batch_size)
    chunk_size=10000000
    chunks_loop = data.shape[0] // chunk_size
    print(data.shape[0]/chunk_size, chunks_loop)
    new_data = np.ndarray((data.shape[0], d))

    for i in range(0, chunks_loop):
        print(i/chunks_loop)
        _temp = pcaObject.partial_fit(data[i*chunk_size : (i+1)*chunk_size, :])
        new_data[i*chunk_size : (i+1)*chunk_size, :] = _temp.fit_transform(data[i*chunk_size : (i+1)*chunk_size, :])
    return new_data

    #X_sparse = sparse.csr_matrix(data)
    #return pcaObject.fit_transform(X_sparse)

def scratchpca(data, d):
    cov_mat = np.cov(data.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) 

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = eigen_pairs[0][1][:, np.newaxis]
    for i in range(1, d):
        w = np.hstack((w, eigen_pairs[i][1][:, np.newaxis]))
    # w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    newData = np.array(data).dot(w)
    # return newData

    pcaObject = PCA(n_components=d)
    reducedData = pcaObject.fit_transform(data)
    return reducedData
    
    
def checkError(data):
    cov_mat = np.cov(data.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) 

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    sorted_eigen_vals = np.array(eigen_pairs, dtype=object)[:,0]
    # print(sorted_eigen_vals)
    totalEigSum = sum(sorted_eigen_vals)
    print("Dimensions\t Error")
    for i in range(1, len(sorted_eigen_vals)):
        error = sum(sorted_eigen_vals[i:])/totalEigSum
        percent_error = round(error*100, 2)
        print(i, "\t\t", '{0:05.2f}'.format(percent_error), "%")


def main():
    print("Performing PCA on Data")


if __name__ == "__main__":
    main()
