from data import data
import numpy as np
import os
import gc
from src import pca
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error as sk_rmsle
from sklearn.ensemble import AdaBoostRegressor
from src import models
from sklearn.preprocessing import StandardScaler

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
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)

    # reduce complexity of data:
    print("Running PCA")
    _mini_train_pca = pca.pca(mini_train, d=3)
    #_mini_train_pca = mini_train
    print("Finished PCA")

    # memory management:
    del mini_train

    _mini_train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(_mini_train_pca, mini_y, test_size=0.1, random_state=42)
    print("Training size: ", _mini_train_pca.shape, "Training label size:", mini_y.shape, "Testing size:", _mini_test_pca.shape, "Testing label size:", mini_test_y.shape)

    # run classifier: regression trees:
    print("Fitting....")
    _model = models.adaBoostRegression(_mini_train_pca, mini_y, _mini_test_pca, mini_test_y)
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
    test_x = np.delete(test_x, np.argwhere(meta=='floor_count'), axis=1)
    # From the sparsity graph, we should probably remove floor count:
    test_x = np.delete(test_x, np.argwhere(meta=='year_built'), axis=1)
    pca_test_x = pca.incremental_pca(test_x, 3, 3000)
    del test_x
    meta = np.delete(meta, np.argwhere(meta=='year_built'))
    meta = np.delete(meta, np.argwhere(meta=='floor_count'))

    test_result = data.test(_model, pca_test_x, is_scipy=True)
    #np.save('test_out_example.npy', test_result)
    data.test_to_csv(test_result,'./submissions/test_adaboost_v1_pca.csv')
    