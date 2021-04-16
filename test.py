# -*- coding: utf-8 -*-

'''
    Author : Choi Hey Min
             
    Title : EXCEL -> PLOT (Tensorflow version2)
    
    최종 수정일 : 2021.04.07
    
    1. line 32 EST 수정
    2. line 59 findDate = datetime.datetime(2019, 8, 11, 12, 0) 날짜 수
    
'''

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

EST = 7


TODAY = datetime.datetime.now().strftime('%Y%m%d')
VER = 'version 1'
PICKLE = '2007to2019.pickle'
      

def getModelAccuracy(real, est):
      
    x = real.values.flatten()
    y = est.values.flatten()
    
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)
    # mpe = np.mean((x-y)/x)*100
    
    print('R2: {} RMSE: {} MAE: {} ' .format(r2, rmse, mae))
    
    
    return r2, rmse, mae


def getRealData():
    ''' get indexDate and real sst_data '''
    
    findDate = datetime.datetime(2019, 8, 10, 12, 0)
    dateList = nc_time.tolist()
    
    for date in dateList:
        if date == findDate:
            indexDate = dateList.index(date)
            break
    
    nc_real = nc_sst[indexDate]
    
    real = pd.DataFrame(nc_real).values.tolist()
            
    
    return real, indexDate

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('create folder : {}' .format(dir))
    except Exception as e:
        print(e)
        
def read_pickle():
    
    # get pickle
    with open(PICKLE, 'rb') as f:
        nc_sst = pickle.load(f)
        nc_lat = pickle.load(f)
        nc_lon = pickle.load(f)
        nc_time = pickle.load(f)
    
    return nc_sst, nc_lat, nc_lon, nc_time   

def drawPlot(data, date, img_text):
    ''' draw real Map '''
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    plt.rc('font', family='times new roman', size=30)
    
    title = 'Map for 10 Aug 2019 12:00:00 ('+img_text+')'
    plt.title(title, weight='bold', ha='center', pad=20)
    
    map = Basemap(projection='cyl', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat), llcrnrlat=np.nanmin(nc_lat),
                  urcrnrlon=np.nanmax(nc_lon), llcrnrlon=np.nanmin(nc_lon))
    map.drawcoastlines(linewidth=1)
    
    map.drawlsmask(land_color='gray', ocean_color='white', lakes=True)
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.5,color='white')
    
    map.drawmapboundary()
    
    
    x,y = map(nc_lon, nc_lat)
    
    min_sst = 20
    max_sst = 30
    
    levels = np.linspace(min_sst, max_sst, 50)
    img = plt.contourf(x, y, data, levels=levels, cmap='coolwarm', extend='both')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.5)
    
    cbar = plt.colorbar(img, ticks=levels, cax=cax)
    cbar.set_label(label='SST(℃)', labelpad=20)
    ticks = np.arange(20, 31, 1)
    cbar.set_ticks(ticks)
    
    
    plt.savefig(date.strftime('%Y%m%d_')+img_text+'.png', bbox_inches='tight')
    print('File Save: ' + img_text + '.png')
    plt.show()

def getModelAccuracy(real, est):
      
    x = real.values.flatten()
    y = est.values.flatten()
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    mape = mean_absolute_percentage_error(x,y)*100
    
    
    return r2, rmse, mape

def checkFile(file):
    
    try:
        if os.path.isfile(file):
            print('Read '+excel_name)
    
    except IOError:
        print(file + ' is not exist')
     
if __name__ == '__main__':


    nc_sst, nc_lat, nc_lon, nc_time = read_pickle()
    
    real, indexDate = getRealData()
    
    real_list = [[0 for j in range(len(nc_lon))] for i in range(len(nc_lat))]
    
    for i in range(len(nc_lat)):
        for j in range(len(nc_lon)):
            
            data = real[i][j]
            
            if math.isnan(data):
                real_list[i][j] = None
            else:
                real_list[i][j] = real[i][j] - 273.15
    
    date = nc_time[indexDate]
    real = pd.DataFrame(real_list)
    
    excel_name = date.strftime('%Y%m%d_')+'EST'+str(EST)+'.xlsx'
    checkFile(excel_name)
        
    est = pd.read_excel(excel_name, sheet_name='Sheet', header=None, index_col=None)
    
    
    r2, rmse, mape = getModelAccuracy(real, est)
    print('R2: {} RMSE: {} MAPE: {} ' .format(r2, rmse, mape))
    
    drawPlot(real, date, 'real')
    drawPlot(est, date, 'est-'+str(EST)+'day')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    