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

def preprocessBuildingData(data_path:str, fill_na=True):
    _data = pandas.read_csv(data_path)

    # figure out what to do with timestamp column:
    _data['timestamp'] = __convert_date__(_data['timestamp'])
    # normalize meter reading?
    #_data['meter_reading'] = __normalize_data__(_data['meter_reading'])
    if fill_na:
        for col in _data.columns:
            if _data[col].isnull().values.any():
                _data[col] = _data[col].fillna(-1)
    return _data

def preprocessWeatherdata(data_path:str, fill_na=True):
    _data = pandas.read_csv(data_path)
    # convert datetime to unix time:
    _data['timestamp'] = __convert_date__(_data['timestamp'])
    if fill_na:
        for col in _data.columns:
            if _data[col].isnull().values.any():
                _data[col] = _data[col].fillna(-1)
    return _data

def preprocessMeta(data_path:str, fill_na=True):
    _data = pandas.read_csv(data_path)
    str_cats = _data.primary_use.to_list()
    str_cat_legend = set(str_cats)
    _data.primary_use = _data.primary_use.astype('category').cat.codes
    _data['primary_use'] = pandas.Categorical(_data['primary_use'])
    # map NaN to -1?
    #_data['floor_count'] = _data['floor_count'].fillna(-1)
    if fill_na:
        for col in _data.columns:
            if _data[col].isnull().values.any():
                _data[col] = _data[col].fillna(-1)
    return _data, str_cats, str_cat_legend

def mapMetaToTrain(df_tb, df_tm, df_wm, fill_na=True):
    df_tb = df_tb.merge(df_tm, left_on='building_id', right_on='building_id', how='left')
    df_tb = df_tb.merge(df_wm, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how="left")
    if fill_na:
        for col in df_tb.columns:
            if df_tb[col].isnull().values.any():
                df_tb[col] = df_tb[col].fillna(-1)
    _cols = df_tb.columns.to_list() # gives you meter_reading as well

    
    # manually remove site 0 based on ASHRAE feeddback:
    df_tb = df_tb[df_tb['site_id'] != 0]
    # print("Removed Site 0 Data Shape: " + str(df_tb.shape))

    y = df_tb.pop('meter_reading')
        
    # memory reduction:
    for col in df_tb.columns:
        _dtype = df_tb[col].dtype
        if _dtype == np.float64:
            df_tb[col] = df_tb[col].astype(np.float32)
    df_tb['building_id'] = df_tb['building_id'].astype(np.uint16)
    df_tb['meter'] = df_tb['meter'].astype(np.uint8)
    


    return df_tb.to_numpy(), y.to_numpy(), _cols # x,y

def saveCache(np_list:np.ndarray, np_file):
    print("Saving cache file for %s"%(np_file))
    np.save(np_file, np_list)

def loadCache(np_file):
    _item = np.load(np_file, allow_pickle=True)
    return _item

def loadTestFeatures(test_csv, weather_test_csv, meta_csv, fill_na=True):
    _weather = preprocessWeatherdata(weather_test_csv, fill_na=fill_na)
    _meta, _, _ = preprocessMeta(meta_csv, fill_na=fill_na)
    _test = preprocessBuildingData(test_csv)
    _test = _test.merge(_meta, left_on='building_id', right_on='building_id', how='left')
    _test = _test.merge(_weather, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how="left")
    # remove row_id (let's hope they stay in order):
    _test = _test.drop(['row_id'], axis=1)

    # memory reduction:
    for col in _test.columns:
        _dtype = _test[col].dtype
        if _dtype == np.float64:
            _test[col] = _test[col].astype(np.float32)
    _test['building_id'] = _test['building_id'].astype(np.uint16)
    _test['meter'] = _test['meter'].astype(np.uint8)

    if fill_na:
        for col in _test.columns:
            if _test[col].isnull().values.any():
                _test[col] = _test[col].fillna(-1)

    _cols = _test.columns.to_list()
    #y = df_tb.pop('meter_reading')
    return _test.to_numpy(), _cols # x,y

def test(model, test_set, is_scipy:bool):
    # reduce in batches:    
    if is_scipy:
        y = model.predict(test_set)
    return y

def test_to_csv(test_out, csv_file_out):
    # test_out -> meter_readings 
    _row_id = np.arange(start=0, stop=test_out.shape[0])
    print(_row_id)
    with open(csv_file_out, 'w') as csvfile:
        fieldnames = ['row_id', 'meter_reading']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(test_out.shape[0])):
            writer.writerow({'row_id':i, 'meter_reading':test_out[i]}) 
    return 0

# Visualizing data:
def featureSparsity(data, x_labels, outfile="featureSparsity", msg_suppress=False):
    if not msg_suppress:
        print("Generating feature sparsity figure...")
        
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
        fig.write_html("./figures/pca_3d_plot.html")
        fig.show()
        
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x,y,z)
        plt.show()

def plot_knn_k(knn_ks, knn_error):
    fig = plt.figure()
    plt.plot(knn_ks, knn_error)
    plt.gca().invert_xaxis()
    plt.xticks(knn_ks)
    plt.xlabel('k values')
    plt.ylabel('RMSL Error')
    plt.title("KNN K-Values vs RMSLE")
    plt.savefig('./figures/knn_k_vs_rmsel.png')

# temp:
def saveCustomDataset():
    train = np.load('data_cache.npy', allow_pickle=True)
    meta = list(np.load('meta_cache.npy', allow_pickle=True))
    y_out = meta.pop(3)
    meta.append(y_out)

    with open('colab_data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=meta)
        writer.writeheader()

        for i in tqdm(range(train.shape[0])):
            writer.writerow({
                                'building_id':train[i,0],
                                'meter': train[i,1],
                                'timestamp': train[i,2],
                                'site_id': train[i,3],
                                'primary_use': train[i,4],
                                'square_feet': train[i,5],
                                'year_built': train[i,6],
                                'floor_count': train[i,7],
                                'air_temperature': train[i,8],
                                'cloud_coverage': train[i,9],
                                'dew_temperature': train[i, 10],
                                'precip_depth_1_hr': train[i,11],
                                'sea_level_pressure': train[i,12],
                                'wind_direction': train[i,13],
                                'wind_speed': train[i,14],
                                'meter_reading': train[i,15]
                            }) 
    return 0

def saveCustomDataset2():
    data_path = './data/ashrae/'
    meta = list(np.load('meta_cache.npy', allow_pickle=True))
    train_file = os.path.join(data_path, 'train.csv')
    train_meta_file = os.path.join(data_path, 'building_metadata.csv')
    train_weather_file = os.path.join(data_path, 'weather_train.csv') # features for training
    test_file = os.path.join(data_path, 'test.csv')
    test_weather_file = os.path.join(data_path, 'weather_test.csv') # features for testing
    test_x, _ = loadTestFeatures(test_file, test_weather_file, train_meta_file) # no y included; ignoring the column name output cuz I already know it
    meta.pop(3)
    #y_label = meta.pop(3)
    #meta.append(y_label)
    print(meta)
    
    with open('colab_data_test.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=meta)
        writer.writeheader()

        for i in tqdm(range(test_x.shape[0])):
            writer.writerow({
                                'building_id':test_x[i,0],
                                'meter': test_x[i,1],
                                'timestamp': test_x[i,2],
                                'site_id': test_x[i,3],
                                'primary_use': test_x[i,4],
                                'square_feet': test_x[i,5],
                                'year_built': test_x[i,6],
                                'floor_count': test_x[i,7],
                                'air_temperature': test_x[i,8],
                                'cloud_coverage': test_x[i,9],
                                'dew_temperature': test_x[i, 10],
                                'precip_depth_1_hr': test_x[i,11],
                                'sea_level_pressure': test_x[i,12],
                                'wind_direction': test_x[i,13],
                                'wind_speed': test_x[i,14],
                                #'meter_reading': train[i,15]
                            }) 

    # From the sparsity graph, we should probably remove floor count:
    #mini_train = np.delete(mini_train, np.argwhere(meta=='floor_count'), axis=1)
    #meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    
    # From the sparsity graph, we should probably remove floor count:
    #mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)

#saveCustomDataset()
#saveCustomDataset2()