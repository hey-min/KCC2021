# -*- coding: utf-8 -*-

# Draw Graph

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def checkFile(file_list):
    
    for file in file_list:
        try:
            if os.path.isfile(file):
                print('Read : {}' .format(file))
    
        except IOError:
            print(file + ' is not exist')

days = np.arange(1,6,1)

# Time Series Graph
sst_list = []
ex_list = []
for i in range(1,6):
    ex_est = 'excel/EST'+str(i)+'_'+str(0.001)+'_'+str(500)+'.xlsx'
    ex_list.append(ex_est)

checkFile(ex_list)

df_real = pd.read_excel('excel/real.xlsx')
LAT, LON = 13, 38
print(df_real.loc[LAT][LON])
for i in range(len(ex_list)):
    
    file = ex_list[i]
    df = pd.read_excel(file)
    data = df.loc[LAT][LON]
    
    print(file+': '+str(data))
    sst_list.append(data)
     
    
    
    
    

'''

# 1. r2, rmse, mape
# ============= R2 =================
r2 = [0.8826, 0.8107, 0.7257, 0.6437, 0.5813]

plt.figure(figsize=(6, 4))
plt.rc('font', size=10)

plt.ylim(0, 1)
plt.plot(days, r2, 'o-', label='R$^2$')
plt.legend()
plt.xticks(np.arange(1, 6, 1))
plt.xlabel('Prediction interval (days)')
plt.title('Comparison of R$^2$')
    
plt.show()

# ============= RMSE =================
rmse = [0.48, 0.61, 0.74, 0.86, 0.95]

plt.figure(figsize=(6, 4))
plt.rc('font', size=10)

plt.ylim(0, 1.4)
plt.plot(days, rmse, 'o-', label='RMSE')
plt.legend()
plt.xticks(np.arange(1, 6, 1))
plt.xlabel('Prediction interval (days)')
plt.title('Comparison of RMSE')
    
plt.show()

# ============= MAPE =================
mape = [1.35, 1.75, 2.17, 2.61, 2.81]

plt.figure(figsize=(6, 4))
plt.rc('font', size=10)

plt.ylim(0, 6)
plt.plot(days, mape, 'o-', label='MAPE')
plt.legend()
plt.xticks(np.arange(1, 6, 1))
plt.xlabel('Prediction interval (days)')
plt.title('Comparison of MAPE')
    
plt.show()


# 2. F1 Score
# ============= F1 Score =================
f1_score = [0.808, 0.767, 0.722, 0.537, 0.417]

plt.figure(figsize=(6, 4))
plt.rc('font', size=10)

plt.ylim(0, 1)
plt.plot(days, f1_score, 'o-', label='F1 Scores')
plt.legend()
plt.xticks(np.arange(1, 6, 1))
plt.xlabel('Prediction interval (days)')
plt.title('Comparison of F1 Scores')
    
plt.show()

# ============= TPR FPR =================

tpr = [0.90441, 0.81281, 0.7284, 0.44802, 0.31204]
fpr = [0.12308, 0.11322, 0.10598, 0.08069, 0.06787]

plt.figure(figsize=(5, 5))
plt.rc('font', size=10)

for i in range(len(tpr)):
    if i==0:
        plt.text(fpr[i], tpr[i], str(days[i])+'day')
    else:
        plt.text(fpr[i], tpr[i], str(days[i])+'days')
    
plt.scatter(fpr, tpr, marker='s')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('FPR')
plt.ylabel('TPR')
    
plt.show()

'''