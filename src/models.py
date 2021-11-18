import numpy as np
import pandas
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error as sk_rmsle
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import time

# Official test set has NO public ground truth; upload to kaggle to get final result
def runTest():
    raise NotImplementedError()

def runValTest():
    raise NotImplementedError()

def rmsle(y, pred):
    raise NotImplementedError()
    n = y.shape[0]
    return np.sqrt( (1/n) * np.sum( np.power( np.log(y+1) - np.log(pred+1),  ) ) )

# regression models:
def regressionTrees(x,y, test_data, test_data_y):
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(x,y)
    pred = model.predict(test_data)
    gt = test_data_y
    _error = sk_rmsle(pred, gt)
    print("Average Validation RMSLE:", _error)
    return model

def regressionSVM(x,y, test_data, test_data_y):
    model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    model.fit(x,y)
    pred = model.predict(test_data)
    gt = test_data_y
    _error = sk_rmsle(pred, gt)
    print("Average Validation RMSLE:", _error)
    return model

def regressionNeighbors(x,y,test_data,test_data_y, k_size=5):
    _model = neighbors.KNeighborsRegressor(k_size)
    model = _model.fit(x,y)
    pred = model.predict(test_data)
    gt = test_data_y
    _error = sk_rmsle(pred, gt)
    print("Average Validation RMSLE:", _error)
    return model

def regressionNeighborsLoop(x,y,test_data,test_data_y, k_size=5):
    t1 = time.time()
    _model = neighbors.KNeighborsRegressor(k_size)
    model = _model.fit(x,y)
    pred = model.predict(test_data)
    gt = test_data_y
    _error = sk_rmsle(pred, gt)
    t2 = time.time()
    print("Time for fit+test+eval: %f seconds"%(t2-t1))
    print("Average Validation RMSLE:", _error)
    return _error