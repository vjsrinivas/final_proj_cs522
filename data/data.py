import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from tqdm import tqdm
import pandas

# Training meta:
DATA_CAT = [
            'building_id',
            'meter',
            'timestamp',
            'meter_reading'
        ]

WEATHER_DATA_CAT = [
            'site_id',
            'timestamp',
            'air_temperature',
            'cloud_coverage',
            'dew_temperature',
            'precip_depth_1_hr',
            '',
        ]

BUILD_META_CAT = [
            '',
            '',
            '',
            ''
        ]


# Parse data:
def parseCSV(filename:str):
    _ret = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(tqdm(reader)):
            _ret.append(row)
    return _ret


# Parse train.csv:
def parseTrainData(filename:str):
    _ret = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(tqdm(reader)):
            _id = row[0]
            if _id not in row:
                _ret[_id] = row[1:]
        
    return _ret

# Visualizing data:


if __name__ == '__main__':
    #parseCSV('./data/train.csv')
    parseTrainData('./data/train.csv')
