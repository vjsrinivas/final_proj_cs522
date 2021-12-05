import numpy as np
import src.models
import src.pca
import argparse
import os
from data import data

from experiments import proto1, proto2, proto3, proto4, proto5, proto6, proto1_b
from experiments import proto7

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--exp', '-e', default='')
    args.add_argument('--data', '-d', default='./data/ashrae')
    return args.parse_args()

def expRun(exp_name:str, *exp_kwargs):
    if exp_name == 'proto1': # regression tree
        # run prototype stuff here:
        proto1.run1(*exp_kwargs)
    elif exp_name == 'proto1b':
        proto1_b.run1(*exp_kwargs)
    elif exp_name == 'proto2': # svm
        proto2.run1(*exp_kwargs)
    elif exp_name == 'proto3': # knn
        proto3.run1(*exp_kwargs)
    elif exp_name == 'proto4': # nn try 1
        proto4.run1(*exp_kwargs)
    elif exp_name == 'proto5':
        proto5.run1(*exp_kwargs)
    elif exp_name == 'proto6':
        proto6.run1(*exp_kwargs)
    elif exp_name == 'adaboost_v1':
        proto7.run1(*exp_kwargs)
    else:
        raise Exception("Unrecognized experiment name!")

if __name__ == '__main__':
    args = parseArgs()
    expRun(args.exp, args.data)