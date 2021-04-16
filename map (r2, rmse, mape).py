# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:25:22 2021

@author: hey-min choi

MAP - R2, RMSE, MAPE

"""

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

EST = 1
ex_real = 'excel/real.xlsx'
ex_est = 'excel/est'+str(EST)+'_20190810.xlsx'

def readPK():
    
    # get pickle
    with open('2007to2019.pickle', 'rb') as f:
        nc_sst = pickle.load(f)
        nc_lat = pickle.load(f)
        nc_lon = pickle.load(f)
        nc_time = pickle.load(f)
    
    return nc_sst, nc_lat, nc_lon, nc_time

def drawPlot(data, img_text):
    
    
    x = data.values.flatten()
    x = x[~np.isnan(x)]
   
    min = int(np.min(x))
    max = int(np.max(x))
    mean = int(np.mean(x))
    print('Min:{} Max:{} MEAN:{}' .format(min, max, mean))
        
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    plt.rc('font', family='times new roman', size=30)
    
    title = 'Map for 10 Aug 2019 12:00:00 ('+img_text+')'
    plt.title(title, weight='bold', ha='center', pad=20)
    
    map = Basemap(projection='cyl', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)
    map.drawcoastlines(linewidth=1)
    
    map.drawlsmask(land_color='white', ocean_color='white', lakes=True)
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.5,color='white')
    
    map.drawmapboundary()
    
    
    x,y = map(nc_lon, nc_lat)
    
    
    levels = np.linspace(min, max, 50)
    img = plt.contourf(x, y, data, levels=levels, cmap='jet', extend='both')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.5)
    
    cbar = plt.colorbar(img, ticks=levels, cax=cax)
    ticks = np.arange(min, max+1, 0.5)
    cbar.set_ticks(ticks)
    
    
    plt.savefig('img/'+img_text+'.png', bbox_inches='tight')
    print('File Save: ' + 'img/'+img_text+'.png')
    plt.show()

def getModelAccuracy(x, y):
      
    # x = real.values.flatten()
    # y = est.values.flatten()
    
    # x = x[~np.isnan(x)]
    # y = y[~np.isnan(y)]
    
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    mape = mean_absolute_percentage_error(x,y)*100
    
    
    return r2, rmse, mape

def r_squared(r, c):
    
    numerator, denominator = 0, 0
    
    x = df_real[r][c]
    y = df_est[r][c]
    
    numerator = 2**(x - y)
    denominator = 2**(x - real_mean)
    
    r_2 = 1 - (numerator/denominator)
    
    list_r2[r][c] = r_2
    
def rmse(r, c):
    
    x = df_real[r][c]
    y = df_est[r][c]
    
    rmse = 2**(x-y)
    
    list_rmse[r][c] = rmse
    
def mape(r, c):
    
    x = df_real[r][c]
    y = df_est[r][c]
    
    mape = 100 * np.abs((x-y)/x)
    
    list_mape[r][c] = mape
    

def checkFile(file):
    
    try:
        if os.path.isfile(file):
            print('Read '+file)
    
    except IOError:
        print(file + ' is not exist')
        
if __name__ == '__main__':
    

    checkFile(ex_real)
    checkFile(ex_est)
    
    nc_sst, nc_lat, nc_lon, nc_time = readPK()
    
    df_real = pd.read_excel(ex_real).to_numpy()
    df_est = pd.read_excel(ex_est).to_numpy()
    
    x = df_real.flatten()
    x = x[~np.isnan(x)]
    real_mean = np.mean(x)
    
    list_r2 = [[None for j in range(len(nc_lon))] for i in range(len(nc_lat))]
    list_rmse = [[None for j in range(len(nc_lon))] for i in range(len(nc_lat))]
    list_mape = [[None for j in range(len(nc_lon))] for i in range(len(nc_lat))]
    
    for r in range(len(nc_lat)):
        
        for c in range(len(nc_lon)):
            
            cell = df_real[r][c]
            
            if math.isnan(cell):continue
            else:
                rmse(r, c)
                mape(r,c)
    
    
    df_rmse = pd.DataFrame(list_rmse)
    df_rmse.to_excel('excel/est'+str(EST)+'_rmse.xlsx')
    drawPlot(df_rmse, 'est'+str(EST)+'_rmse')
    
    df_mape = pd.DataFrame(list_mape)
    df_mape.to_excel('excel/est'+str(EST)+'_mape.xlsx')
    drawPlot(df_mape, 'est'+str(EST)+'_mape')
    
    
    # 평균 구하기
    # x = df_rmse.values.flatten()
    # y = df_mape.values.flatten()
    
    # x = x[~np.isnan(x)]
    # y = y[~np.isnan(y)]
    
    # list = [x, y]
    # for item in list:
    #     min = np.round(np.min(item), 3)
    #     max = np.round(np.max(item), 3)
    #     mean = np.round(np.mean(item), 3)
    #     print('Min:{} Max:{} MEAN:{}' .format(min, max, mean))
    
    
    
    