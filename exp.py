import numpy as np
import src.models
import src.pca
import src.fld
import argparse
import os
from data import data
from experiments.proto1 import *

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

if __name__ == '__main__':
    args = parseArgs()
    expRun(args.exp, args.data)