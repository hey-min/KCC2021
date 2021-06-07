# -*- coding: utf-8 -*-
# 2021.06.02

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox



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
# df = pd.DataFrame(None, columns=['sst'])
# df['sst'] = nc_sst[0:, LAT, LON]- 273.15


df_time = pd.DataFrame(None, columns=['date'])
df_time['date'] = nc_time
df_time['date_year'] = df_time['date'].dt.year

df_time_input = df_time.loc[df_time['date_year']==2019]   
 
start = int(min(df_time_input.index))
end = int(max(df_time_input.index))

print('Input : {} ~ {}' .format(min(df_time_input['date']), max(df_time_input['date'])))

# find year's date count
find_year_cnt = df_time_input['date'].count()



# model test
time_est = np.zeros(find_year_cnt)
time_label = np.zeros(find_year_cnt)

est = np.zeros(find_year_cnt-EST)
label = np.zeros(find_year_cnt-EST)


def get_data(lat, lon):
    
    total_size = len(nc_time)
    columns = ['sst']
    df = pd.DataFrame(columns=columns)
    
    for i in range(total_size):
        new_sst = np.round(nc_sst[i, lat, lon] - 273.15, 4)
        df_data = {'sst':new_sst}
        df = df.append(pd.Series(df_data, index=df.columns), ignore_index=True)
    
    return df




######################################################################
####################  CLASS :  WindowGenerator   #####################
######################################################################


class WindowGenerator():
    
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

   
    
    def plot(self, model=None, plot_col='sst', max_subplots=1):
        
        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            if model is not None:
                predictions = model(inputs)


        data_input = inputs.numpy()
        data_output = predictions.numpy()
        
        return data_input, data_output


    def make_dataset(self, data):
        
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=30,)

        ds = ds.map(self.split_window)
      
        return ds


@property
def train(self):
  # print('TRAIN START')
  return self.make_dataset(self.train_df)

@property
def val(self):
  # print('VALIDATION START')
  return self.make_dataset(self.val_df)

@property
def test(self):
  # print('TEST START')
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.test))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example



MAX_EPOCHS = IT
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(lr=LR),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      verbose=2,
                      callbacks=[early_stopping])
  return history



def plot_scatter():
    
    x = np.array(rslt_real)
    y = np.array(rslt_est)
    
    denominator = x.dot(x) - x.mean() * x.sum()
    m = ( x.dot(y) - y.mean() * x.sum() ) / denominator
    b = ( y.mean() * x.dot(x) - x.mean() * x.dot(y) ) / denominator
    y_pred = m*x + b
    
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
        
    plt.rc('font', size=25)
    
    title = 'Scatter Diagram for 2019 (Pixel ['+str(LAT)+']['+str(LON)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.xlim([5, 30])
    plt.ylim([5, 30])
    plt.xticks([10,15,20,25,30])
    plt.yticks([10,15,20,25,30])
    
    plt.scatter(rslt_real, rslt_est, c='red', s=5)
    plt.rcParams['lines.linewidth'] = 1
    plt.plot(x, y_pred, 'blue')
    
    plt.xlabel('Real SST', labelpad=20)
    
    if EST==1:
        plt.ylabel('Estimated SST('+str(EST)+'-day)', labelpad=20)
    else:
        plt.ylabel('Estimated SST('+str(EST)+'-days)', labelpad=20)
        
    text = 'R$^2$: ' + str(r2) + '  RMSE: '+ str(rmse) +'  MAPE: ' + str(mape)
    text2 = '\nLR: ' + str(LR) + '  IT: ' + str(IT)
    textbox = offsetbox.AnchoredText(text+text2, loc='upper left')
    ax.add_artist(textbox)
    
    #### SAVE IMAGE ####
    img_file = 'scatter_'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()    
    


def plot_timeSeries(): 

    plt.figure(figsize=(15, 12))
    ax = plt.gca()
        
    plt.rc('font', size=25)

    title = 'Time Series Graph for 2019 (Pixel ['+str(LAT)+']['+str(LON)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.plot(rslt_est, label='EST-'+str(EST)+'Day(s) SST', color='blue')
    plt.plot(rslt_real, label='Label', color='gray')
    plt.legend(loc='lower right')
    plt.xlabel('Days')
    plt.ylabel("SST(℃)")
    plt.axhline(y=28, color='r', linewidth=1) 
    plt.grid()
    plt.yticks(np.arange(10, 35, 5))
    
    
    #### SAVE IMAGE ####
    img_file = 'timeSeries_'+str(EST)+'_'+str(LR)+'_'+str(IT)+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()


def accuracy():
    
    df_real = rslt_real
    df_est = rslt_est
    
    for i in range(len(df_real)):
    
        real = df_real[i]
        est = df_est[i]
        df_real[i] = round(real, 4)
        df_est[i] = round(est, 4)


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
        
        if TP == 0: break
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



    
if __name__ == '__main__':
    
    
    
    model_path = 'model_test/'+VER
    model_name = '['+str(LAT)+']['+str(LON)+']'
    
    print('MODEL:{}' .format(model_path+'/'+model_name+'.h5'))
    new_model = tf.keras.models.load_model(model_path+'/'+model_name+'.h5')
    
    print('est:{} lat:{} lon:{} lr:{} it:{}' .format(EST, LAT, LON, LR, IT))
  

    df = get_data(LAT, LON)


    train_df = df[0:3653]
    val_df = df[3653:4383]

    rslt_real = []
    rslt_est = []
    
    for i in range(start, end+1):
    
        LABEL = i
        END = LABEL - EST
        START = END - 30 + 1
    
    
        test_df = df[START:LABEL+1]



        wide_window = WindowGenerator(
            input_width=30, label_width=30, shift=EST,
            label_columns=['sst'],
            train_df=train_df, val_df=val_df, test_df=test_df)


    
    
    

        ds_test = wide_window.test

        list_test = list(ds_test.as_numpy_iterator())

        rslt_input = list_test[0][0]
        rslt_label = list_test[0][1]


        rslt_input, rslt_output = wide_window.plot(new_model)

    
        est_sst = np.round(rslt_output[0][29][0], 4)
    
        real_sst = np.round(rslt_label[0][29][0], 4)
    
        print('Est: {} Real: {} ' .format(est_sst, real_sst))
        
        rslt_real.append(real_sst)
        rslt_est.append(est_sst)

    
    df_rslt = accuracy()
    print(df_rslt)
    
    r2 = round(df_rslt['R2'][0], 3)
    rmse = round(df_rslt['RMSE'][0],3)
    mape = round(df_rslt['MAPE'][0],3)
    
    plot_timeSeries()
    
    plot_scatter()

















