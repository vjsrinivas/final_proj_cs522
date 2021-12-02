from math import e
import os
import numpy as np
from data import data 
from src import pca
from src import models
from sklearn.model_selection import train_test_split
import gc
from sklearn.preprocessing import StandardScaler
from src import kernalpca

# Due 11/18/2021:
def run1(data_path):
    train_file = os.path.join(data_path, 'train.csv')
    train_meta_file = os.path.join(data_path, 'building_metadata.csv')
    train_weather_file = os.path.join(data_path, 'weather_train.csv') # features for training
    test_file = os.path.join(data_path, 'test.csv')
    test_weather_file = os.path.join(data_path, 'weather_test.csv') # features for testing

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
    
    '''
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    '''

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
    print("Running PCA")
    _mini_train_pca = pca.pca(mini_train, d=3)
    # _mini_train_pca = kernalpca.kernalpca(mini_train, d=3)
    print("Finished PCA")

    # memory management:
    del mini_train

    _mini_train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(_mini_train_pca, mini_y, test_size=0.1, random_state=42)
    print("Training size: ", _mini_train_pca.shape, "Training label size:", mini_y.shape, "Testing size:", _mini_test_pca.shape, "Testing label size:", mini_test_y.shape)

    '''
    print("PCA on training data")
    if _mini_train_pca.shape[0] > 100000:
        data.pca_3d_plot(_mini_train_pca[:100000,:])
    else:
        data.pca_3d_plot(_mini_train_pca)
    '''

    # run classifier: regression trees:
    print("Fitting....")
    _model = models.regressionTrees(_mini_train_pca, mini_y, _mini_test_pca, mini_test_y)
    del _mini_train_pca
    del mini_y
    del _mini_test_pca
    del mini_test_y
    del mini_train_idx
    gc.collect(1)

    # test on real testset:
    # reduce with PCA:
    print("Reading in testset...")
    test_x, _ = data.loadTestFeatures(test_file, test_weather_file, train_meta_file) # no y included; ignoring the column name output cuz I already know it
    pca_test_x = pca.incremental_pca(test_x, 3, 3000)
    del test_x
    test_result = data.test(_model, pca_test_x, is_scipy=True)
    np.save('test_out_example.npy', test_result)
    data.test_to_csv(test_result,'./submissions/test_proto3.csv')
    