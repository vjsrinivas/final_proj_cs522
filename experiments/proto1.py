import os
import numpy as np
from data import data 

# Due 11/18/2021:
def prototype1(data_path):
    # run data processing for training, testing etc. here:
    train_file = os.path.join(data_path, 'train.csv')
    train_meta_file = os.path.join(data_path, 'building_metadata.csv')
    train_weather_file = os.path.join(data_path, 'weather_train.csv')
    
    train_building_data = data.preprocessBuildingData(train_file)
    train_meta_data, str_cats, str_cat_legend = data.preprocessMeta(train_meta_file)
    train_weather_data = data.preprocessWeatherdata(train_weather_file)

    # combine data frames together:
    train_data = data.combineDataFrames(train_building_data, train_meta_data, train_weather_data) 
    print(train_data)

    # reduce complexity of data:

    # run classifier: MPP Case 1?