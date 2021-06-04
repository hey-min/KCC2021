# -*- coding: utf-8 -*-

# Draw Graph

import matplotlib.pyplot as plt
import numpy as np



days = np.arange(1,6,1)

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


# 2. f1 score
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

