# -*- coding: utf-8 -*-

# 2021.06.07

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import datetime
import os
import argparse
import warnings

import wide_window



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--EST', type=int, default=1, dest='EST')
parser.add_argument('--LAT', type=int, default=13, dest='LAT')
parser.add_argument('--LON', type=int, default=38, dest='LON')
parser.add_argument('--LR', type=float, default=0.001, dest='LR')
parser.add_argument('--IT', type=int, default=500, dest='IT')

args = parser.parse_args()

EST = args.EST
LAT = args.LAT
LON = args.LON
LR = args.LR
IT = args.IT
VER = str(EST)+'_'+str(LR)+'_'+str(IT)

# hyper parameter
YEAR = 365
SIZE_BATCH = 30 
N_STEPS = 30


NAME_PICKLE = '2007to2019.pickle'
with open(NAME_PICKLE, 'rb') as f:
    nc_sst = pickle.load(f)
    nc_lat = pickle.load(f)
    nc_lon = pickle.load(f)
    nc_time = pickle.load(f)
    
total_size = len(nc_time)


# ================== 2019 DataFrame =========================
df = pd.DataFrame(None, columns=['sst'])
df['sst'] = nc_sst[0:, LAT, LON]- 273.15


df_time = pd.DataFrame(None, columns=['date'])
df_time['date'] = nc_time
df_time['date_year'] = df_time['date'].dt.year

df_input = df_time.loc[df_time['date_year']==2019]    
start = int(min(df_input.index))
end = int(max(df_input.index))

print('Input : {} ~ {}' .format(min(df_input['date']), max(df_input['date'])))

# find year's date count
find_year_cnt = df_input['date'].count()



# model test
time_est = np.zeros(find_year_cnt)
time_label = np.zeros(find_year_cnt)

est = np.zeros(find_year_cnt-EST)
label = np.zeros(find_year_cnt-EST)


# ============== MODEL TEST ===============  
for i in df_input.index:
        
    id_end = i - EST
    id_start = id_end - SIZE_BATCH + 1
    
    input_size = id_end - id_start + 1
    
    print('INPUT SIZE: ', input_size)
    print('{}day-EST : {}~{}'.format(EST, id_start, id_end))
    
    # time_est[i] = ValTest(id_start, id_end)
    model_path = 'model_test/'+VER
    model_name = '['+str(LAT)+']['+str(LON)+']'
    
    print('est:{} lat:{} lon:{} lr:{} it:{}' .format(EST, LAT, LON, LR, IT))
  

    model_path = model_path+'/'+model_name+'.h5'
    print('MODEL:{}' .format(model_path))
    
    model = tf.keras.models.load_model(model_path)
    
    input = df.loc[id_start:id_end]['sst'].to_frame()
    
    print('\n-------------------MODEL TEST-------------------')
    ds_test = wide_window.test
    model.evaluate(ds_test, verbose=2)
    est_sst = model.evaluate(input)

    list_test = list(ds_test.as_numpy_iterator())

    rslt_input = list_test[0][0]
    rslt_label = list_test[0][1]


    rslt_input, rslt_output = wide_window.plot(new_model)
    
    est_sst = rslt_output[0][29]
    
    print(est_sst)

for i in range(SIZE_BATCH-EST):
        
    est[i] = time_est[i+EST]
    label[i] = time_label[i]    
    

    
    
    
    
    
    