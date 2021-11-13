import numpy as np
import models
import reducer
import argparse
import os
<<<<<<< HEAD
from data import data
=======
>>>>>>> 9e7dba54434fa3908fd3a2d549364653650fa2c3

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--exp', '-e', default='')
    args.add_argument('--data', '-d', default='./data/ashrae')
    return args.parse_args()

def expRun(exp_name:str, *exp_kwargs):
    if exp_name == 'proto1':
        # run prototype stuff here:
<<<<<<< HEAD
        prototype1(*exp_kwargs)
    else:
        raise Exception("Unrecognized experiment name!")

=======
        pass
    else:
        raise Exception("Unrecognized experiment name!")

# All the experiments we want to do with different classifiers and stuff:

>>>>>>> 9e7dba54434fa3908fd3a2d549364653650fa2c3
# Due 11/18/2021:
def prototype1(data_path):
    # run data processing for training, testing etc. here:
    train_file = os.path.join(data_path, 'train.csv')
<<<<<<< HEAD
    data.preprocessBuildingData(train_file)
=======
>>>>>>> 9e7dba54434fa3908fd3a2d549364653650fa2c3

if __name__ == '__main__':
    args = parseArgs()
    expRun(args.exp, args.data)