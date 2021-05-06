# -*- coding: utf-8 -*-
"""
    choi hey-min
    
    test.py
    
    2021.05.06

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import warnings
import os
import math
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore')

TODAY = datetime.datetime.now().strftime('%Y%m%d')
VER = 'version 1'
PICKLE = '2007to2019.pickle'
  

# Hyper parameter

EST = 1
IT = 500
LR = 0.001


    
def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('create folder : {}' .format(dir))
    except Exception as e:
        print(e)

def checkFile(file):
    
    try:
        if os.path.isfile(file):
            print('Read '+file)
    
    except IOError:
        print(file + ' is not exist')
        


# 1. 예측 sst excel file 확인
excel_name = 'EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.xlsx'
checkFile(excel_name)       
est = pd.read_excel(excel_name, sheet_name='Sheet', header=None, index_col=None)
est = est.drop([0])        
        

# 2. 데이터 정리
PICKLE = '2007to2019.pickle'
with open(PICKLE, 'rb') as f:
    nc_sst = pickle.load(f)
    nc_lat = pickle.load(f)
    nc_lon = pickle.load(f)
    nc_time = pickle.load(f)
    
df = pd.DataFrame(None, columns=['date'])
df['date'] = nc_time
df['date_year'] = df['date'].dt.year

# 기준 날짜 설정
date_of_real = datetime.datetime(2019, 8, 10, 12, 0)
# 기준 날짜의 index
index_of_real = df.loc[df['date']==date_of_real].index[0]
# 기준 날짜의 sst
nc_real = nc_sst[index_of_real]    
    
real_list = [[0 for j in range(len(nc_lon))] for i in range(len(nc_lat))]
# 기준 날짜의 sst k->c
for i in range(len(nc_lat)):
    for j in range(len(nc_lon)):
            
        data = nc_real[i][j]
            
        if math.isnan(data):
            real_list[i][j] = None
        else:
            real_list[i][j] = nc_real[i][j] - 273.15
    
real = pd.DataFrame(real_list)    




def drawPlot(data, date, img_text):
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    plt.rc('font', family='times new roman', size=30)
    
    title = 'Map for 10 Aug 2019 12:00:00 ('+img_text+')'
    plt.title(title, weight='bold', ha='center', pad=20)
    
    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)
    
    
    map.drawcoastlines(linewidth=0.8)
    
    # map.drawlsmask(land_color='grey', ocean_color='white', lakes=True)
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
    
    # text1 = 'Predict Date: '+ str(date)
    text2 = 'LR: '+str(LR)+' IT: '+str(IT)
    text3 = 'R$^2$: '+str(r2)+' RMSE: '+str(rmse)+' MAPE: '+str(mape)
    
    text = text2 + '\n' + text3
    
    textbox = offsetbox.AnchoredText(text, loc='upper left', prop=dict(size=30))
    ax.add_artist(textbox)
    
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
    
    cmap = plt.get_cmap('coolwarm')
    img = map.pcolormesh(x, y, data, cmap=cmap, shading='gouraud')
   
    
    min_sst = 20
    max_sst = 30
    
    levels = np.linspace(min_sst, max_sst, 50)
    # img = plt.contourf(x, y, data, levels=levels, cmap='coolwarm', extend='both')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.5)
    
    cbar = plt.colorbar(img, ticks=levels, cax=cax, extend='both')
    cbar.set_label(label='SST(℃)', labelpad=20)
    ticks = np.arange(20, 31, 1)
    cbar.set_ticks(ticks)
    
    
    plt.savefig('EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png', bbox_inches='tight')
    print('File Save: ' + img_text + '.png')
    plt.show()




def accuracy(real, est):
      
    x = real.values.flatten()
    y = est.values.flatten()
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    r2 = np.round(r2_score(x, y), 3)
    rmse = np.round(np.sqrt(mean_squared_error(x, y)), 3)
    mape = np.round(mean_absolute_percentage_error(x,y)*100, 3)
    
    return r2, rmse, mape




# 3. real과 est의 정확도 r2, rmse, mape
r2, rmse, mape = accuracy(real, est)
print('R2: {} RMSE: {} MAPE: {} ' .format(r2, rmse, mape))


# 4. draw plot  
# drawPlot(real, date_of_real, 'real')
drawPlot(est, date_of_real, 'est-'+str(EST)+'day')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    