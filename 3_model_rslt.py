# -*- coding: utf-8 -*-
# 2021.06.02

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
import datetime
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore')

TODAY = datetime.datetime.now().strftime('%Y%m%d')
VER = 'version 1'
PICKLE = '2007to2019.pickle'
  

# ===== Hyper parameter =====

parser = argparse.ArgumentParser()
parser.add_argument('--EST', type=int, default=7, dest='EST')
parser.add_argument('--LR', type=float, default=0.001, dest='LR')
parser.add_argument('--IT', type=int, default=500, dest='IT')
args = parser.parse_args()


EST = args.EST
LR = args.LR
IT = args.IT



# ===== CHECK FILE =====
def checkFile(file_list):
    
    for file in file_list:
        try:
            if os.path.isfile(file):
                print('Read : {}' .format(file))
    
        except IOError:
            print(file + ' is not exist')
        


# ===== EXCEL FILE : EST AND REAL =====
# ===== CHECK EXCEL FILE
ex_est = 'excel/EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.xlsx'
ex_real= 'excel/real.xlsx'

ex_list = [ex_est, ex_real]
checkFile(ex_list)

df_real = pd.read_excel(ex_real).stack().values.tolist()
df_est = pd.read_excel(ex_est).stack().values.tolist()    
# est = pd.read_excel(excel_name, sheet_name='Sheet', header=None, index_col=None)
# est = est.drop([0])        
        

# ===== READ PICKLE =====
PICKLE = '2007to2019.pickle'
with open(PICKLE, 'rb') as f:
    nc_sst = pickle.load(f)
    nc_lat = pickle.load(f)
    nc_lon = pickle.load(f)
    nc_time = pickle.load(f)



def drawPlot(ex_data):
    
    df = pd.read_excel(ex_data)
    x = df.values.flatten()
    x = x[~np.isnan(x)]
    
    min = int(np.min(x))
    max = int(np.max(x))
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    plt.rc('font', size=25)
    
    title = 'Map for 10 Aug 2019 12:00'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.title(str(ex_data), loc='right', pad=20)
    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)

    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
    
    
    text = 'R$^2$: '+str(r2)+'  RMSE: '+str(rmse)+'  MAPE: '+str(mape)
    text2 = 'F1 Score: ' + str(f1_score)
    text_rslt = text + '\n' + text2
    
    textbox = offsetbox.AnchoredText(text_rslt, loc='upper left')
    ax.add_artist(textbox)

    
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
        
    cmap = plt.get_cmap('coolwarm')
    img = map.pcolormesh(x, y, df, cmap=cmap, shading='gouraud')
    img.set_clim(vmin=20, vmax=30)
    
    levels = np.arange(20, 31, 1)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
        
    cbar = plt.colorbar(img, ticks=levels, cax=cax, extend='both')
    cbar.set_label(label='SST(℃)', labelpad=20)
    
    #### SAVE IMAGE ####
    img_file = 'EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()




def accuracy():
    
    for i in range(len(df_real)):
    
        real = df_real[i]
        est = df_est[i]
        df_real[i] = round(real, 2)
        df_est[i] = round(est, 2)


    y_true = df_real
    y_pred = df_est

    # R2, RMSE, MAPE
    R2 = 0
    RMSEa = 0
    MAPEa = 0
    k = 0
    for k in range(len(df_real)):
        
        Xlist = y_true
        Ylist = y_pred
        x = np.array(y_true)
        y = np.array(y_pred)
   
        res = x - y
        tot = x - y.mean()   
        R2 = 1 - np.sum(res**2) / np.sum(tot**2)
        
        RMSE = np.sqrt(((y-x) ** 2).mean())
    
        MAPE = np.mean(np.abs((x - y)/x)) * 100

    columns = ['EST', 'R2', 'RMSE', 'MAPE']

    data_ac = {'EST':[EST],
        'R2':[round(R2,4)],
        'RMSE':[round(RMSE,2)],
        'MAPE':[round(MAPE,2)]}
    
    df_ac = pd.DataFrame(data_ac, index=data_ac['EST'])
    
    # TP, FP, FN, TN
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(df_real)):
        if y_true[i] > 28 and y_pred[i] > 28:
            TP = TP + 1
        if y_true[i] < 28 and y_pred[i]> 28:
            FP = FP + 1
        if y_true[i] > 28 and y_pred[i] < 28:
            FN = FN + 1
        if y_true[i] < 28 and y_pred[i] < 28:
            TN = TN + 1


    # precision, recall, accuracy, f1 score
    TPR = 0
    FPR = 0
    Precision = 0
    Recall = 0
    F1 = 0
    Accuracy = 0

    for i in range(len(df_real)):
        
        TPR = TP / (TP+FN)
        FPR = FP / (FP+TN)
        
        Precision = TP / (TP+FP)
        Recall = TP / (TP+FN)
        F1 = 2*(Precision * Recall) / (Precision + Recall)
        Accuracy = (TP + TN) / (TP + FN + FP + TN)


    columns = ['EST', 'accuracy', 'recall', 'precision', 'f1_score', 'TPR', 'FPR']

    data_hwt = {'EST':[EST], 'accuracy':[round(Accuracy,2)],
        'recall':[round(Recall,2)],
        'precision':[round(Precision,2)],
        'f1_score':[round(F1,5)],
        'TPR':[round(TPR,5)],
        'FPR':[round(FPR,5)]}

    df_hwt = pd.DataFrame(data_hwt, index=data_hwt['EST'])
    
    df_rslt = pd.merge(df_ac, df_hwt, on='EST')
    
    
    return df_rslt

    

# ===== DRAW EST AND REAL DIFF PLOT =====    

def plot_diff():
    
    df_real = pd.read_excel(ex_real)
    df_est = pd.read_excel(ex_est)
    df = df_real - df_est
    
    x = df.values.flatten()
    x = x[~np.isnan(x)]
    
    _min = abs(min(x))
    _max = abs(max(x))
    
    _abs = 3
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
        
    plt.rc('font', size=25)
        
    title = 'Error Map for 10 Aug 2019 12:00'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.title(str(ex_est), loc='right', pad=20)
    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)

    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
        
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
        
    cmap = plt.get_cmap('jet')
    img = map.pcolormesh(x, y, df, cmap=cmap, shading='gouraud')
    img.set_clim(vmin=-_abs, vmax=_abs)
        
    levels = np.arange(-_abs, _abs+1, 0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
        
    cbar = plt.colorbar(img, ticks=levels, cax=cax, extend='both')
    cbar.set_label(label='SST(℃)', labelpad=20)
        
        
    #### SAVE IMAGE ####
    img_file = 'diff_EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()
    

    return df

   
    
    
def plot_diff_abs():
    
    df_real = pd.read_excel(ex_real)
    df_est = pd.read_excel(ex_est)
    df = df_real - df_est
    
    df = df.abs()
    
    x = df.values.flatten()
    # remove nan values from array
    x = x[~np.isnan(x)]
    
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
        
    plt.rc('font', size=25)
        
    title = 'Positive Error Map for 10 Aug 2019 12:00'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.title(str(ex_est), loc='right', pad=20)
    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)

    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
        
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
        
    cmap = plt.get_cmap('jet')
    img = map.pcolormesh(x, y, df, cmap=cmap, shading='gouraud')
    img.set_clim(vmin=0, vmax=3)
        
    levels = np.arange(0, 4, 0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
        
    cbar = plt.colorbar(img, ticks=levels, cax=cax, extend='both')
    cbar.set_label(label='SST(℃)', labelpad=20)
        
        
    #### SAVE IMAGE ####
    img_file = 'diff_abs_EST'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()
    

    return df




# ===== GET MODEL ACCURACY =====
df_rslt = accuracy()

r2 = df_rslt['R2'][0]
rmse = df_rslt['RMSE'][0]
mape = df_rslt['MAPE'][0]
f1_score = round(df_rslt['f1_score'][0], 3)

print(df_rslt)


# ===== DRAW EST MAP PLOT =====
drawPlot(ex_est)

# ===== DRAW DIFF MAP PLOT =====
df_diff = plot_diff() 

# ===== DRAW DIFF ABS MAP PLOT =====
df_diff_abs = plot_diff_abs()        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    