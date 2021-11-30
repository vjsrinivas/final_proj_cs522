from data import data
import numpy as np
import os
import gc
from src import pca
from src import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def run1(data_path, lr=0.01, epochs=100, minibatch=8, network_shape=[13,20,30,4,1]):
    x, y, _test = loadTrainData(data_path)
    model = nn.Network(network_shape)
    y /= np.max(y)

    _train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(x, y, test_size=0.1, random_state=42)
    
    print(_train_pca.shape, _mini_test_pca.shape, mini_y.shape, mini_test_y.shape)
    mini_y = np.expand_dims(mini_y, axis=1)
    mini_test_y = np.expand_dims(mini_test_y, axis=1)
    _train = []
    _test = []
    print(_train_pca.shape, _mini_test_pca.shape, mini_y.shape, mini_test_y.shape)
    exit()
    model.SGD(_train_pca, epochs, minibatch, lr, mini_test_y)
    #print(sample, mini_y[i])

    return 0

def loadTrainData(data_path):
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
    
    gc.collect()

    meta = np.delete(meta, 3) # remove "meter_reading from meta labels"
    mini_size = x.shape[0] # force full lengths
    #mini_size = 1000000 # cap training set
    np.random.seed(10)
    mini_train_idx = np.random.choice(x.shape[0], size=mini_size)
    mini_train, mini_y = x[mini_train_idx], y[mini_train_idx]
    print("Mini training size:", mini_train.shape, mini_y.shape)

    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='floor_count'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='year_built'))

    # reduce complexity of data:
    _mini_train_pca = pca.pca(mini_train, d=3)

    # memory management:
    del mini_train
    
    print("Testing")
    test_x, _ = data.loadTestFeatures(test_file, test_weather_file, train_meta_file) # no y included; ignoring the column name output cuz I already know it
    pca_test_x = pca.incremental_pca(test_x, 3, 3000)
    del test_x
    
    return _mini_train_pca, y, pca_test_x # train_x, train_y, test_x