import numpy as np
import pandas
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error as sk_rmsle

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
    print("Average RMSLE:", _error)