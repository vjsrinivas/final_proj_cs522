from numpy import linalg
import torch
import torch.nn as nn
from data import data
import numpy as np
import os
import gc
from src import pca
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, dataloader
from collections import OrderedDict
from sklearn.metrics import mean_squared_log_error as sk_rmsle
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class ASHRAEDataset(Dataset):
    def __init__(self, x, y, use_cuda=False, no_y=False) -> None:
        super().__init__()
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.no_y = no_y

        if use_cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        
        if not no_y:
            assert self.x.shape[0] == self.y.shape[0], "X and Y should be the same shape!"
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.no_y:
            return self.x[index].float()
        else:    
            return (self.x[index].float(), self.y[index].float() )

def run1(data_path, lr=0.001, epochs=25, momentum=0.9):
    x, y, _test = loadTrainData(data_path)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    _test = scaler.transform(_test)

    #scaler_int = np.linalg.norm(y)
    #y = y / scaler_int
    
    train, test, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

    model = tf.keras.Sequential([
        Dense(13, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        #Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        #Dropout(0.2),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),)
    #            metrics=tf.keras.losses.MeanSquaredError())
    
    '''
    re_model = keras.models.load_model('experiments/mlp', compile=False)
    '''
    
    model.fit(train, train_y, epochs=4, batch_size = 512, validation_data=(test, test_y), validation_batch_size=64)
    
    # validation set RMSLE:
    val_preds = model.predict(test, batch_size=512)
    _val = np.sqrt(sk_rmsle(test_y, val_preds))
    print("Validation error:", _val)

    model.save('mlp')

    _preds = model.predict(_test, batch_size=512)
    _preds = np.squeeze(_preds)
    data.test_to_csv(_preds, 'submissions/test_proto6_keras.csv')

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
    #meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    
    # From the sparsity graph, we should probably remove floor count:
    mini_train = np.delete(mini_train, np.argwhere(meta=='year_built'), axis=1)
    #meta = np.delete(meta, np.argwhere(meta=='year_built'))
    
    print("Testing")
    test_x, _ = data.loadTestFeatures(test_file, test_weather_file, train_meta_file) # no y included; ignoring the column name output cuz I already know it
    test_x = np.delete(test_x, np.argwhere(meta=='floor_count'), axis=1)
    test_x = np.delete(test_x, np.argwhere(meta=='year_built'), axis=1)
    
    meta = np.delete(meta, np.argwhere(meta=='floor_count'))
    meta = np.delete(meta, np.argwhere(meta=='year_built'))

    return mini_train, y, test_x # train_x, train_y, test_x