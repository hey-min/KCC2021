# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:51:06 2021

@author: Kiost

KCC2021 LSTM Model Main

"""


import pickle
import os
import math
from openpyxl import Workbook, load_workbook
import datetime
import numpy as np
import pandas as pd

# cmd = 'python train_ver_lstm.py --EST=1 --LAT=0 --LON=5'
# print(cmd)
# os.system(cmd)

NAME_PICKLE = '2007to2019.pickle'  

def read_pickle():
    
    # get pickle
    with open(NAME_PICKLE, 'rb') as f:
        nc_sst = pickle.load(f)
        nc_lat = pickle.load(f)
        nc_lon = pickle.load(f)
        nc_time = pickle.load(f)
    
    return nc_sst, nc_lat, nc_lon, nc_time

      
if __name__ == '__main__':
    
    nc_sst, nc_lat, nc_lon, nc_time = read_pickle()
    
    # 1. real data to excel
    # findDate = datetime.datetime(2019, 8, 10, 12, 0)
    # dateList = nc_time.tolist()
    
    # for date in dateList:
    #     if date == findDate:
    #         indexDate = dateList.index(date)
    #         break
    
    # data = nc_sst[indexDate]
    
    
    
    # for r in range(len(nc_lat)):
    #     for c in range(len(nc_lon)):
    #         if math.isnan(data[r,c]):continue
    #         else:data[r][c] = data[r,c]-273.15
    
    # df_real = pd.DataFrame(data)
    
    
    # excel_name = 'real('+str(findDate)[:10]+').xlsx'
    # df_real.to_excel(excel_name, index=False)
    
    
    
    
    # 1. cmd to train.py
    for r in range(len(nc_lat)):
        for c in range(len(nc_lon)):
        
            data = nc_sst[0][r][c]
            # print(data)
            if math.isnan(data):
                # print('-')
                continue
            else:
                cmd = 'python train.py --EST=1 --LAT='+str(r)+' --LON='+str(c) + ' --LR=0.0005 --IT=500'
                # cmd = 'python test.py 1'
                # print(cmd)
                print(os.system(cmd))
