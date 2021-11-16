import os
import numpy as np
from data import data 
from src import pca

# Due 11/18/2021:
def prototype1(data_path):
    # run data processing for training, testing etc. here:
    if os.path.exists('xy_cache.npy'):
        xy = data.loadCache('xy_cache.npy')
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
        x,y = data.mapMetaToTrain(train_building_data, train_meta_data, train_weather_data)
        data.saveCache( np.concatenate([x,np.expand_dims(y,axis=1)], axis=1), 'xy_cache.npy')

    # reduce complexity of data:

    # run classifier: MPP Case 1?
