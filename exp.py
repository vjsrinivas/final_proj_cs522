import numpy as np
import models
import reducer
import argparse
import os

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--exp', '-e', default='')
    args.add_argument('--data', '-d', default='./data/ashrae')
    return args.parse_args()

def expRun(exp_name:str, *exp_kwargs):
    if exp_name == 'proto1':
        # run prototype stuff here:
        pass
    else:
        raise Exception("Unrecognized experiment name!")

# All the experiments we want to do with different classifiers and stuff:

# Due 11/18/2021:
def prototype1(data_path):
    # run data processing for training, testing etc. here:
    train_file = os.path.join(data_path, 'train.csv')

if __name__ == '__main__':
    args = parseArgs()
    expRun(args.exp, args.data)