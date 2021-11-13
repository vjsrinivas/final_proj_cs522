import numpy as np
import models
import reducer
import argparse
import os
from data import data

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--exp', '-e', default='')
    args.add_argument('--data', '-d', default='./data/ashrae')
    return args.parse_args()

def expRun(exp_name:str, *exp_kwargs):
    if exp_name == 'proto1':
        # run prototype stuff here:
        prototype1(*exp_kwargs)
    else:
        raise Exception("Unrecognized experiment name!")

# Due 11/18/2021:
def prototype1(data_path):
    # run data processing for training, testing etc. here:
    train_file = os.path.join(data_path, 'train.csv')
    data.preprocessBuildingData(train_file)

if __name__ == '__main__':
    args = parseArgs()
    expRun(args.exp, args.data)