# -*- coding: utf-8 -*-

# 2021.06.02


import pickle
import os
import math
from openpyxl import Workbook, load_workbook
import datetime
import numpy as np
import pandas as pd

NAME_PICKLE = '2007to2019.pickle'  
with open(NAME_PICKLE, 'rb') as f:
    nc_sst = pickle.load(f)
    nc_lat = pickle.load(f)
    nc_lon = pickle.load(f)
    nc_time = pickle.load(f)

    
def train():
    
    for r in range(len(nc_lat)):
        for c in range(len(nc_lon)):
        
            data = nc_sst[0][r][c]
            # print(data)
            if math.isnan(data):
                # print('-')
                continue
            else:
                cmd = 'python 1_model_train.py --EST='+EST+' --LAT='+str(r)+' --LON='+str(c) + ' --LR='+LR+' --IT='+IT
                print(os.system(cmd))
      
if __name__ == '__main__':
    
 
    est_list = [3]
    LR = str(0.0001)
    IT = str(10000)
    
    for est in est_list :
        
        EST = str(est)
        
        try:
            train()
            print('Train Complete')
    
        except Exception as ex:
            print('Error {}' .format(ex))
    
        else:
            # 2. output :  excel file
            # create predict sst excel file by using created h5 model file
            cmd = 'python 2_model_val_test.py --EST='+EST+' --LR='+LR+' --IT='+IT
            print(cmd)
            print(os.system(cmd))
            
            # 3. output : accuracy test results [r2, rmse, mape, f1 score, TPR, FPR]
            # calculate model accuracy test by using created excel file
            # cmd = 'python test_v2.py --EST='+EST+' --LR='+LR+' --IT='+IT
            # print(cmd)
            # print(os.system(cmd))
