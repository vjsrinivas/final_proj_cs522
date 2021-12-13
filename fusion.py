import numpy as np
from sklearn.metrics import mean_squared_log_error as sk_rmsle
from sklearn.metrics import r2_score as sk_r2score
from data import data

val_model_1 = np.load('val_fuse_1.npy',allow_pickle=True)
val_model_2 = np.load('val_fuse_2.npy', allow_pickle=True)
val_gt = np.load('val_gt.npy', allow_pickle=True)
test_model_1 = np.load('test_1.npy', allow_pickle=True)
test_model_2 = np.load('test_2.npy', allow_pickle=True)

val_model_1 = np.expand_dims(val_model_1, axis=1)
val_model_2 = np.expand_dims(val_model_2, axis=1)
p_fuse_val = np.concatenate((val_model_1, val_model_2), axis=1)
print(p_fuse_val.shape)
#print(p_fuse_val[0])
val_result = np.mean(p_fuse_val, axis=1)
#print(val_result[0])
print(val_result.shape)

print("rmsle:",sk_rmsle(val_gt, val_result))
print("r2:",sk_r2score(val_gt, val_result))

test_model_1 = np.expand_dims(test_model_1, axis=1)
test_model_2 = np.expand_dims(test_model_2, axis=1)
fused_test = np.concatenate((test_model_1, test_model_2), axis=1)
test_result = np.mean(fused_test, axis=1)
data.test_to_csv(test_result, 'submissions/fusion_regression_adaboost.csv')