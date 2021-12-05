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
    #scaler_train_y = np.linalg.norm(train_y)
    #train_y = train_y / scaler_train_y

    #scaler_test_y = np.linalg.norm()

    #for i in range(train.shape[1]):
    #    train_norm[:,i] = train_norm[:,i] / np.linalg.norm(train[:,i])
    
    #for i in range(_test.shape[1]):
    #    _test[:,i] = _test[:,i] / np.linalg.norm(_test[:,i])

    # extra preprocessing done here:
    _train_pca = ASHRAEDataset(train, train_y, use_cuda=True)
    _val_pca = ASHRAEDataset(test, test_y, use_cuda=True)
    _test_pca = ASHRAEDataset(_test, [], use_cuda=True, no_y=True)
    print("Train size:", len(_train_pca))
    print("Val size:", len(_val_pca))
    print("Test size:", len(_test_pca))
    
    train_loader = DataLoader(_train_pca, batch_size=1000000)
    val_loader = DataLoader(_val_pca, batch_size=1000000)
    test_load = DataLoader(_test_pca, batch_size=1000000)

    # define model
    model = createModel(13,1) # full dimensions
    
    if os.path.exists('nn_proto6.pt'):
        model.load_state_dict( torch.load('nn_proto6.pt') )
        model = model.cuda()
    else:
        #model = createSmallestModel()
        opt = torch.optim.SGD(model.parameters(), lr, momentum)
        loss = nn.MSELoss()
        #loss = RMSELoss()

        model = model.cuda()
        loss = loss.cuda()
        model.train()
        print(model)

        for epoch in range(epochs):
            print("Epoch %i"%(epoch))
            avg_epoch_loss = 0
            for i,(x,y) in enumerate(tqdm(train_loader)):
                opt.zero_grad()
                preds = model(x)
                
                #for j in range(x.shape[0]):
                    #print(preds[j], x[j])
                #print(y[0], y[1000], y[12000])
                #preds *= scaler_int
                #preds = torch.squeeze(preds)
                y = torch.unsqueeze(y, axis=1)
                #_loss = torch.sqrt( loss(y, preds) )
                _loss = loss(y, preds)
                _loss.backward()
                avg_epoch_loss = (_loss + avg_epoch_loss)/(i+1)
                opt.step()
            print("Average loss: %f"%(avg_epoch_loss))
            print("====================")
            #input()

            print("Validating at epoch %i..."%(epoch))
            avg_rmsle = 0
            with torch.no_grad():
                model.eval()
                for i, (x,y) in enumerate(tqdm(val_loader)):
                    pred = model(x)
                    y = torch.unsqueeze(y, axis=1)
                    #pred = torch.squeeze(pred)
                    #pred *= scaler_int
                    _pred = pred.detach().cpu().numpy()
                    _y = y.detach().cpu().numpy()
                    #_pred *= scaler_int
                    #_y *= scaler_int
                    _error = np.sqrt(sk_rmsle(_pred, _y))
                    #print(_error)
                    avg_rmsle += _error
            #print(pred[0])
            print("Average RMSLE: %f"%(avg_rmsle/i))

            model.train()

        torch.save(model.state_dict(), 'nn_proto6.pt')

    _out = np.ndarray((len(_test_pca)))
    with torch.no_grad():
        for i, x in enumerate(tqdm(test_load)):
            #print(x)
            _pred = model(x)
            _out[i*1000000:(i+1)*1000000] = np.squeeze(_pred.cpu().numpy())

    data.test_to_csv(_out, 'submissions/test_proto6.csv')

def createModel():
    _model = nn.Sequential(OrderedDict([
        ('dense_1', nn.Linear(13,256) ),
        ('relu_1', nn.ReLU()),
        ('dense_2', nn.Linear(256,256) ),
        ('relu_2', nn.ReLU()),
        ('dense_3', nn.Linear(256,128) ),
        ('relu_3', nn.ReLU()),
        ('dense_4', nn.Linear(128,64) ),
        ('relu_4', nn.ReLU()),
        ('dense_5', nn.Linear(64,32) ),
        ('relu_5', nn.ReLU()),
        ('dense_6', nn.Linear(32,16) ),
        ('relu_6', nn.ReLU()),
        ('dense_7', nn.Linear(16,8) ),
        ('relu_7', nn.ReLU()),
        ('dense_8', nn.Linear(8,4) ),
        ('relu_8', nn.ReLU()),
        ('dense_9', nn.Linear(4,2) ),
        ('relu_9', nn.ReLU()),
        ('dense_10', nn.Linear(2,1) ),
    ]))

    return _model

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