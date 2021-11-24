from math import e
import os
import numpy as np
from numpy.lib.function_base import average
from data import data 
from src import pca
from src import models
from sklearn.model_selection import train_test_split
import gc
from sklearn.model_selection import KFold

# Due 11/18/2021:
def run1(data_path):
    train_file = os.path.join(data_path, 'train.csv')
    train_meta_file = os.path.join(data_path, 'building_metadata.csv')
    train_weather_file = os.path.join(data_path, 'weather_train.csv') # features for training
    test_file = os.path.join(data_path, 'test.csv')
    test_weather_file = os.path.join(data_path, 'weather_test.csv') # features for testing
    do_kfold_testing = True

    _data_cache = 'data_cache.npy'
    _meta_cache = 'meta_cache.npy'

    # run data processing for training, testing etc. here:
    if os.path.exists(_data_cache):
        print("Reading from caching...")
        xy = data.loadCache(_data_cache)
        meta = data.loadCache(_meta_cache)
        x = xy[:, :15]
        y = xy[:, 15]
    else:
        print("WARNING: Cache not found. Generating from original dataset")
        print("Reading training csv: %s ..."%train_file)
        train_building_data = data.preprocessBuildingData(train_file)
        print("Reading meta data csv: %s ..."%train_meta_file)
        train_meta_data, str_cats, str_cat_legend = data.preprocessMeta(train_meta_file)
        print("Reading weather data csv: %s ..."%train_weather_file)
        train_weather_data = data.preprocessWeatherdata(train_weather_file)

        # combine data frames together:
        print("Combining into one dataframe...")
        x,y,meta = data.mapMetaToTrain(train_building_data, train_meta_data, train_weather_data)
        
        # CACHE results to save time from reprocessing files again and again and again and again...
        # Delete cache if you need to change something about the preprocessing!
        print("Caching...")
        data.saveCache( np.concatenate([x,np.expand_dims(y,axis=1)], axis=1), _data_cache)
        data.saveCache(meta, _meta_cache)
    
    gc.collect()

    meta = np.delete(meta, 3) # remove "meter_reading from meta labels"
    mini_size = x.shape[0] # force full lengths
    #mini_size = 1000000 # cap training set
    np.random.seed(10)
    mini_train_idx = np.random.choice(x.shape[0], size=mini_size)
    mini_train, mini_y = x[mini_train_idx], y[mini_train_idx]
    print("Mini training size:", mini_train.shape, mini_y.shape)

    # Graph out how sparse every feature is:
    #data.featureSparsity(mini_train, meta) # disable fill nan
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='floor_count'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='year_built'))
    
    # remove all rows with NaN:
    '''
    _mask = ~np.isnan(mini_train).any(axis=1)
    mini_train = mini_train[_mask, :]
    mini_y = mini_y[_mask]
    assert mini_train.shape[0] == mini_y.shape[0]
    '''

    # reduce complexity of data:
    _mini_train_pca = pca.pca(mini_train, d=3)

    # memory management:
    del mini_train

    # _mini_train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(_mini_train_pca, mini_y, test_size=0.1, random_state=42)
    # print("Training size: ", _mini_train_pca.shape, "Training label size:", mini_y.shape, "Testing size:", _mini_test_pca.shape, "Testing label size:", mini_test_y.shape)

    # run classifier: regression trees:
    _knn_rmsle = []

    if do_kfold_testing:
        print("Fitting....")
        k_fold = 10
        kf = KFold(n_splits=k_fold)
        for k in [1,5,10,15,20,25,30]:
            print()
            print("--------------------------------"+str(k)+"--------------------------------")
            print()
            k_fold_error = []
            run = 0
            for train_index, test_index in kf.split(_mini_train_pca):
                run += 1
                print("Run " + str(run))
                # print("Train Length: " + str(len(train_index)) + " Test Size: " + str(len(test_index)))
                X_train, X_test = _mini_train_pca[train_index], _mini_train_pca[test_index]
                y_train, y_test = mini_y[train_index], mini_y[test_index]

                _error = models.regressionNeighborsLoop(X_train, y_train, X_test, y_test, k_size=k)
                k_fold_error.append(_error)

            _knn_rmsle.append(np.average(np.array(k_fold_error)))
            print(str(k_fold) + "-Fold Cross Validation Errors: " + str(k_fold_error))
            print("Average Error for " + str(k_fold) + "-Fold Cross validation: " + str(np.average(np.array(k_fold_error))))
        np.save('knn_proto3_output.npy', _knn_rmsle)
        # _knn_rmsle = np.load('knn_proto3_output.npy', allow_pickle=True)
        data.plot_knn_k([1,5,10,15,20,25,30], _knn_rmsle)
    else:
        print("Not doing k-fold. Assuming we've selected best k value.")
        _mini_train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(_mini_train_pca, mini_y, test_size=0.1, random_state=42)
        print("Training size: ", _mini_train_pca.shape, "Training label size:", mini_y.shape, "Testing size:", _mini_test_pca.shape, "Testing label size:", mini_test_y.shape)

        print("Running test...")
        print("Reading in testset...")
        _model = models.regressionNeighbors(_mini_train_pca, mini_y, _mini_test_pca, mini_test_y, k_size=1)
        print("Fitting")
        test_x, _ = data.loadTestFeatures(test_file, test_weather_file, train_meta_file) # no y included; ignoring the column name output cuz I already know it
        pca_test_x = pca.incremental_pca(test_x, 3, 3000)
        del test_x
        test_result = data.test(_model, pca_test_x, is_scipy=True)
        data.test_to_csv(test_result,'./submissions/test_proto3.csv')