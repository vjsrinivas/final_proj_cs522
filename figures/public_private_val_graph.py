import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig = plt.figure()
    method_names = ['Random Forest', 'AdaBoost (Regression Tree)','Regression Tree','MLP','kNN', 'Linear Regression']
    X_axis = np.arange(len(method_names))
    public_error = [1.363, 1.491, 1.502, 2.239, 2.704, 3.792]
    private_error = [1.758, 1.873, 1.8929, 2.306, 3.098, 4.340]
    val_error = [0.8, 1.329, 1.337, 2.2, 1.5, 3.795]
    plt.bar(X_axis, public_error, 0.2, label='Public RMSLE')
    plt.bar(X_axis + 0.2, private_error, 0.2, label='Private RMSLE')
    plt.bar(X_axis - 0.2, val_error, 0.2, label='Validation RMSLE')
    plt.xticks(X_axis, method_names, rotation=45) 
    plt.xlabel('Method Name')
    plt.ylabel('RMSLE Score (lower the better)')
    plt.title('Public vs Private Test Set vs Validation RMSLE')
    plt.tight_layout()
    plt.legend(['Public','Private', 'Validation'])
    plt.savefig('public_vs_private_vs_val_rmsle.png')
    plt.show()