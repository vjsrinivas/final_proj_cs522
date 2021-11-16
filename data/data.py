import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from tqdm import tqdm
import pandas

# Training meta:
DATA_CAT = [
            'building_id',
            'meter',
            'timestamp',
            'meter_reading'
        ]

WEATHER_DATA_CAT = [
            'site_id',
            'timestamp',
            'air_temperature',
            'cloud_coverage',
            'dew_temperature',
            'precip_depth_1_hr',
            '',
        ]

BUILD_META_CAT = [
            '',
            '',
            '',
            ''
        ]

def __convert_date__(col):
    date_col = pandas.to_datetime(col)
    np_col = date_col.to_numpy()
    np_date = np_col.astype('datetime64[s]').astype('int')
    return np_date

def __normalize_data__(col):
    _max_value = col.max()
    col /= _max_value
    return col

def preprocessBuildingData(data_path:str):
    _data = pandas.read_csv(data_path)

    # figure out what to do with timestamp column:
    _data['timestamp'] = __convert_date__(_data['timestamp'])
    # normalize meter reading?
    _data['meter_reading'] = __normalize_data__(_data['meter_reading'])
    return _data

def preprocessWeatherdata(data_path:str):
    _data = pandas.read_csv(data_path)
    # convert datetime to unix time:
    _data['timestamp'] = __convert_date__(_data['timestamp'])
    return _data

def preprocessMeta(data_path:str):
    _data = pandas.read_csv(data_path)
    str_cats = _data.primary_use.to_list()
    str_cat_legend = set(str_cats)
    _data.primary_use = _data.primary_use.astype('category').cat.codes
    _data['primary_use'] = pandas.Categorical(_data['primary_use'])
    # map NaN to -1?
    #_data['floor_count'] = _data['floor_count'].fillna(-1)
    return _data, str_cats, str_cat_legend

def combineDataFrames(df_b, df_m, df_w):
    cats = []
    for df in [df_b, df_m, df_w]:
        cats.append(df.columns.to_list())
    df_concat = pandas.concat([df_b, df_m, df_w])
    return df_concat.to_numpy(), cats

def combineMetaBuilding(df_b, df_m):
    cats = []
    for df in [df_b, df_m]:
        cats.append(df.columns.to_list())
    df_concat = pandas.concat([df_b, df_m])
    return df_concat.to_numpy(), cats

def mapMetaToTrain(df_tb, df_tm, df_wm):
    df_tb = df_tb.merge(df_tm, left_on='building_id', right_on='building_id', how='left')
    df_tb = df_tb.merge(df_wm, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how="left")
    y = df_tb.pop('meter_reading')
    return df_tb.to_numpy(), y.to_numpy() # x,y

def mapMetatoTest(df_test, df_tm):
    return 0

def saveCache(np_list:np.ndarray, np_file):
    print("Saving cache file for %s"%(np_file))
    np.save(np_file, np_list)

def loadCache(np_file):
    _item = np.load(np_file, allow_pickle=True)
    return _item

# Visualizing data:

