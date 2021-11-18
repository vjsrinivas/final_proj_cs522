import os
import numpy as np
from data import data 
from src import pca

# Due 11/18/2021:
def prototype1(data_path):
    _data_cache = 'data_cache.npy'
    _meta_cache = 'meta_cache.npy'

    # run data processing for training, testing etc. here:
    if os.path.exists(_data_cache):
        xy = data.loadCache(_data_cache)
        meta = data.loadCache(_meta_cache)
        x = xy[:, :15]
        y = xy[:, 15]
    else:
        train_file = os.path.join(data_path, 'train.csv')
        train_meta_file = os.path.join(data_path, 'building_metadata.csv')
        train_weather_file = os.path.join(data_path, 'weather_train.csv')
        
        train_building_data = data.preprocessBuildingData(train_file)
        train_meta_data, str_cats, str_cat_legend = data.preprocessMeta(train_meta_file)
        train_weather_data = data.preprocessWeatherdata(train_weather_file)

        # combine data frames together:
        x,y,meta = data.mapMetaToTrain(train_building_data, train_meta_data, train_weather_data)
        data.saveCache( np.concatenate([x,np.expand_dims(y,axis=1)], axis=1), _data_cache)
        data.saveCache(meta, _meta_cache)

    print(x[0], y[0], meta)
    meta = np.delete(meta, 3) # remove "meter_reading from meta labels"
    mini_size = 100000
    mini_train_idx = np.random.choice(x.shape[0], size=mini_size)
    mini_train, mini_y = x[mini_train_idx], y[mini_train_idx]
    print(mini_train.shape)
    print(mini_y.shape)
    data.featureSparsity(mini_train, meta)
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='floor_count'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)
    meta = np.delete(meta, np.argwhere(meta=='year_built'))
    
    # remove all rows with NaN:
    _mask = ~np.isnan(mini_train).any(axis=1)
    mini_train = mini_train[_mask, :]
    print(mini_train.shape)

    # reduce complexity of data:
    _mini_train_pca = pca.pca(mini_train, d=3)
    print(_mini_train_pca.shape)
    data.pca_3d_plot(_mini_train_pca)

    # run classifier: MPP Case 1?
