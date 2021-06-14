# -*- coding: utf-8 -*-
# 2021.06.10


import os
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from openpyxl import Workbook, load_workbook

tf.compat.v1.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--EST', type=int, default=1, dest='EST')
parser.add_argument('--LAT', type=int, default=0, dest='LAT')
parser.add_argument('--LON', type=int, default=5, dest='LON')
parser.add_argument('--LR', type=float, default=0.001, dest='LR')
parser.add_argument('--IT', type=int, default=500, dest='IT')

args = parser.parse_args()


EST = args.EST
LAT = args.LAT
LON = args.LON
LR = args.LR
IT = args.IT
VER = str(EST)+'_'+str(LR)+'_'+str(IT)

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        print(e)
        

model_path = 'model_month/'+VER
createFolder(model_path)
model_name = '['+str(LAT)+']['+str(LON)+']'

NAME_PICKLE = '2007to2019.pickle'
   

def read_pickle():
    
    # get pickle
    with open(NAME_PICKLE, 'rb') as f:
        nc_sst = pickle.load(f)
        nc_lat = pickle.load(f)
        nc_lon = pickle.load(f)
        nc_time = pickle.load(f)
    
    return nc_sst, nc_lat, nc_lon, nc_time


def get_data(lat, lon):
    
    total_size = len(nc_time)
    columns = ['sst']
    df = pd.DataFrame(columns=columns)
    
    for i in range(total_size):
        new_sst = np.round(nc_sst[i, lat, lon] - 273.15, 4)
        # new_date= str(nc_time[i])[:10]
        df_data = {'sst':new_sst}
        df = df.append(pd.Series(df_data, index=df.columns), ignore_index=True)
    
    return df



print(VER)

nc_sst, nc_lat, nc_lon, nc_time = read_pickle()

df = get_data(LAT, LON)

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:3653]
val_df = df[3653:4383]

# find_date = '2019-08-10 12:00:00'
# df_date = pd.DataFrame(nc_time, columns={'date'})
# find_index = df_date[df_date['date']==find_date].index.values

# print('\n{} : {}' .format(find_date, find_index))

# _index = int(find_index)
# _id_end = int(_index - EST)
# _id_start = int(_id_end - 30 + 1)
# test_df = df[_id_start:_index+1]
test_df = df[4383:]
           



######################################################################
####################  CLASS :  WindowGenerator   #####################
######################################################################


class WindowGenerator():
    
    def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
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
        f'\nTotal window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}\n'])



def split_window(self, features):
    
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
        
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='sst', max_subplots=1):
    
    inputs, labels = self.example
    # plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        # plt.subplot(3, 1, n+1)
        # plt.ylabel(f'{plot_col} [normed]')
        # plt.plot(self.input_indices, inputs[n, :, plot_col_index],
        #      label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        # plt.scatter(self.label_indices, labels[n, :, label_col_index],
        #         edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            # plt.scatter(self.label_indices, predictions[n, :, label_col_index],
            #       marker='X', edgecolors='k', label='Predictions',
            #       c='#ff7f0e', s=64)

        # if n == 0:
        #     plt.legend()

    # plt.xlabel('Time [h]')
    
    data_labels = labels.numpy()
    data_output = predictions.numpy()
        
    return data_labels, data_output

WindowGenerator.plot = plot


def make_dataset(self, data):
        # (input_window, label_window)

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=EST,
            shuffle=True,
            batch_size=30,)

        ds = ds.map(self.split_window)
      
        return ds
WindowGenerator.make_dataset = make_dataset    
    
@property
def train(self):
  print('TRAIN START')
  return self.make_dataset(self.train_df)

@property
def val(self):
  print('VALIDATION START')
  return self.make_dataset(self.val_df)

@property
def test(self):
  print('TEST START')
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

@property
def example_test(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example_test', None)
  if result is None:
    # No example batch was found, so get one from the `.test` dataset
    result = next(iter(self.test))
    # And cache it for next time
    self._example_test = result
  return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
WindowGenerator.example_test = example_test



def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(lr=LR),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=IT,
                      validation_data=window.val,
                      verbose=0,
                      callbacks=[early_stopping])
  return history



wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=EST,
    label_columns=['sst'])
print(wide_window)



lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)


history = compile_and_fit(lstm_model, wide_window)
    
wide_window.plot(lstm_model)



# ========== MODEL SAVE AND LOAD ============

lstm_model.save(model_path+'/'+model_name+'.h5')
print('Model Save : '+model_path+'/'+model_name+'.h5')

# new_model = tf.keras.models.load_model(model_path+'/'+model_name+'.h5')



# print('MODEL:{}' .format(model_path+'/'+model_name+'.h5'))
# new_model = tf.keras.models.load_model(model_path+'/'+model_name+'.h5')

# ds_test = wide_window.test
# new_model.evaluate(ds_test, verbose=2)
# list_test = list(ds_test.as_numpy_iterator())
# rslt_input = list_test[0][0]
# rslt_label = list_test[0][1]

        
# rslt_label, rslt_output = wide_window.plot(new_model)

# ============== LSTM MODEL PREDICT AND DRAW PLOT ==============
# inputs, labels = wide_window.example_test
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plot_col_index = wide_window.column_indices['sst']
# plt.title('Model Test Results')
# plt.ylabel('sst')
# plt.plot(wide_window.input_indices, inputs[0, :, 0], label='Inputs', marker='.', zorder=-10)
# plt.scatter(wide_window.label_indices, labels[0, :, 0], edgecolors='k', label='Labels', c='#2ca02c', s=64)
# predictions = new_model(labels)
# plt.scatter(wide_window.label_indices, predictions[0, :, 0], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
# plt.legend()
# plt.xlabel('Time [h]')

# predictions_numpy = predictions.numpy()
# rslt = predictions_numpy[0,29,0]
# diff = labels.numpy()-predictions.numpy()
# print(diff[0,29,0])




