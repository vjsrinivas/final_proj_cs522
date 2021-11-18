PLOTLY = True

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from tqdm import tqdm
import pandas
if PLOTLY:
    import plotly.express as px
    import plotly.graph_objects as go

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
    _cols = df_tb.columns.to_list()
    y = df_tb.pop('meter_reading')
    return df_tb.to_numpy(), y.to_numpy(), _cols # x,y

def mapMetatoTest(df_test, df_tm):
    return 0

def saveCache(np_list:np.ndarray, np_file):
    print("Saving cache file for %s"%(np_file))
    np.save(np_file, np_list)

def loadCache(np_file):
    _item = np.load(np_file, allow_pickle=True)
    return _item

# Visualizing data:
def featureSparsity(data, x_labels, outfile="featureSparsity"):
    num_na = []
    fig = plt.figure(figsize=(10,5))
    plt.title("Sparsity of Features Present")
    for i in range(data.shape[1]):
        _na = np.count_nonzero( np.isnan(data[:,i]) ) / data.shape[0]
        #print(_na)
        num_na.append(_na)
    #print(len(num_na), len(x_labels))
    plt.barh(x_labels, num_na)
    plt.ylabel("Feature Type")
    plt.xlabel("Sparsity Percentage")
    plt.savefig(os.path.join('figures', "%s.png"%outfile))

def pca_3d_plot(data):
    x,y,z = data[:,0], data[:,1], data[:,2]
    if PLOTLY:
        marker_data = go.Scatter3d(
            x=data[:,0], 
            y=data[:,1], 
            z=data[:,2], 
            marker=go.scatter3d.Marker(size=3), 
            opacity=0.8, 
            mode='markers'
        )
        fig=go.Figure(data=marker_data)
        fig.show()
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x,y,z)
        plt.show()