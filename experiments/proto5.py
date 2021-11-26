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

class ASHRAEDataset(Dataset):
    def __init__(self, x, y, use_cuda=False) -> None:
        super().__init__()
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        if use_cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        assert self.x.shape[0] == self.y.shape[0], "X and Y should be the same shape!"
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index].float(), self.y[index].float() )

def run1(data_path, lr=0.001, epochs=100, momentum=0.9):
    if not os.path.exists('data_pca_3d_cache.npy'):
        x, y, _test = loadTrainData(data_path)
        np.save('data_pca_3d_cache.npy', np.concatenate( (x, np.expand_dims(y, axis=1)), axis=1 ) )
        np.save('data_test_pca_3d_cache.npy', _test)
    else:
        xy = np.load('data_pca_3d_cache.npy', allow_pickle=True)
        x, y = xy[:,:-1], xy[:,-1]
        if os.path.exists('data_test_pca_3d_cache.npy'):
            _test = np.load('data_test_pca_3d_cache.npy', allow_pickle=True)
        else: 
            x, y, _test = loadTrainData(data_path)

    _train_pca, _mini_test_pca, mini_y, mini_test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # extra preprocessing done here:
    _train_pca = ASHRAEDataset(_train_pca, mini_y, use_cuda=True)
    _val_pca = ASHRAEDataset(_mini_test_pca, mini_test_y, use_cuda=True)
    print("Train size:", len(_train_pca))
    print("Val size:", len(_val_pca))
    
    train_loader = DataLoader(_train_pca, batch_size=100000)
    val_loader = DataLoader(_val_pca, batch_size=100000)

    # define model
    #model = createModel(13,1) # full dimensions
    model = createSmallestModel()
    opt = torch.optim.SGD(model.parameters(), lr, momentum)
    loss = nn.MSELoss()

    model = model.cuda()
    loss = loss.cuda()
    model.train()
    print(model)

    for epoch in range(epochs):
        print("Epoch %i"%(epoch))
        avg_epoch_loss = 0
        for i,(x,y) in tqdm(enumerate(train_loader)):
            opt.zero_grad()
            preds = model(x)
            print(preds)
            preds = torch.squeeze(preds)
            _loss = loss(y, preds)
            _loss.backward()
            avg_epoch_loss = (_loss + avg_epoch_loss)/(i+1)
            opt.step()
        print("Average loss: %f"%(avg_epoch_loss))

        print("Validating at epoch %i..."%(epoch))
        avg_rmsle = 0
        with torch.no_grad():
            model.eval()
            for i, (x,y) in tqdm(enumerate(val_loader)):
                pred = model(x)
                pred = torch.squeeze(pred)
                print(pred)
                _error = sk_rmsle(pred.cpu().numpy(), y.cpu().numpy())
                avg_rmsle += _error
        print(pred[0])
        print("Average RMSLE: %f"%(avg_rmsle/i))

        model.train()

def createModel(in_dim, out_dim):
    _model = nn.Sequential(OrderedDict([
        ('dense_1', nn.Linear(in_dim, in_dim//2) ),
        ('relu_1', nn.LeakyReLU()),
        ('dense_2', nn.Linear(in_dim//2, in_dim//4)),
        ('relu_2', nn.LeakyReLU()),
        ('dense_3', nn.Linear(in_dim//4, in_dim//6)),
        ('relu_3', nn.LeakyReLU()),
        ('dense_4', nn.Linear(in_dim//6, out_dim)),
        ('relu_4', nn.LeakyReLU())
    ]))
    return _model

def createSmallestModel(in_dim=3, out_dim=1):
    _model = nn.Sequential(OrderedDict([
        ('dense_1', nn.Linear(in_dim, 5) ),
        ('relu_1', nn.LeakyReLU()),
        ('dense_2', nn.Linear(5, 2) ),
        ('relu_2', nn.LeakyReLU()),
        ('dense_3', nn.Linear(2,1)),
        ('relu_3', nn.LeakyReLU())
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