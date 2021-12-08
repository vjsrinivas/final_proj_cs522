import numpy as np
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt

def rmsle(pred, truth):
    return np.sqrt(mean_squared_log_error(truth, pred))

if __name__ == '__main__':
    # plot constants with RMSLE:
    fig = plt.figure()
    constants = [i for i in range(0,500)]
    gt = np.array([50])
    _error = []
    
    for i in constants:
        pred = np.array([i])
        _error.append(rmsle(gt, pred))
   
    plt.plot(constants, _error)
    plt.scatter(gt, _error[gt[0]], c='r')
    plt.legend(['Prediction RMSLE Error', 'Ground Truth'])
    plt.title("RMSLE Error vs Constant Prediction")
    plt.xlabel('Constant Value')
    plt.ylabel('RMSLE Value')
    plt.tight_layout()
    plt.savefig('rmsle_error_metric.png')