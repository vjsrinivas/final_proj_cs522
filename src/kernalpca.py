from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA

def kernalpca(data, d):
    X = data
    transformer = KernelPCA(n_components=d)
    X_transformed = transformer.fit_transform(X)
    return X_transformed
