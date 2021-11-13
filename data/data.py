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

def preprocessBuildingData(data_path:str):
    _data = pandas.read_csv(data_path)
    print(_data[:, "timestamp"])
    
    return 0

def preprocessWeatherdata(data_path:str):
    _data = pandas.read_csv(data_path)

# Visualizing data:

