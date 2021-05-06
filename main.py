# -*- coding: utf-8 -*-

"""
    choi hey-min
    
    main.py
    
    2021.05.06
    
"""


import pickle
import os
import math


NAME_PICKLE = '2007to2019.pickle'  
with open(NAME_PICKLE, 'rb') as f:
    nc_sst = pickle.load(f)
    nc_lat = pickle.load(f)
    nc_lon = pickle.load(f)
    nc_time = pickle.load(f)

      
if __name__ == '__main__':

    for r in range(len(nc_lat)):
        for c in range(len(nc_lon)):
        
            data = nc_sst[0][r][c]

            if math.isnan(data):

                continue
            else:
                cmd = 'python train.py --EST=1 --LAT='+str(r)+' --LON='+str(c) + ' --LR=0.0005 --IT=500'

                print(os.system(cmd))
