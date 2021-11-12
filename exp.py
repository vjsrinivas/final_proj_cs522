import numpy as np
import models
import reducer
import argparse

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--exp', '-e', default='')
    args.add_argument('--data', '-d', default='./data/ashrae')
    return args.parse_args()

if __name__ == '__main__':
    args = parseArgs()
