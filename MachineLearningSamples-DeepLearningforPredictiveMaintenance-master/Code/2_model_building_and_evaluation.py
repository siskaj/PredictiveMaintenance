
# coding: utf-8

# # Step 2: Model Building & Evaluation
# Using the training and test data sets we constructed in the `Code/1_data_ingestion_and_preparation.ipynb` Jupyter notebook, this notebook builds a LSTM network for scenerio described at [Predictive Maintenance Template](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) to predict failure in aircraft engines. We will store the model for deployment in an Azure web service which we build in the `Code/3_operationalization.ipynb` Jupyter notebook.

# In[1]:


import keras

# import the libraries
import h5py
import os
import pandas as pd
import numpy as np

import urllib
import glob
import pickle
import re

from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from sklearn import datasets
from keras.layers import Dense, Dropout, LSTM, Activation

# Use the Azure Machine Learning data collector to log various metrics
#from azureml.logging import get_azureml_logger
#run_logger = get_azureml_logger()
#run_logger.log('amlrealworld.predictivemaintenanceforpm.modelbuildingandevaluation','true')


# ## Load feature data set
# 
# We have previously created the labeled data set in the `Code\1_Data Ingestion and Preparation.ipynb` Jupyter notebook and stored it in local persistant storage. We define the storage locations for both the notebook input and output here.

# In[4]:


# We will store each of these data sets in a local persistance folder
SHARE_ROOT = '../../Data/'

# These file names detail the data files. 
TRAIN_DATA = 'PM_train_files.pkl'
TEST_DATA = 'PM_test_files.pkl'

# We'll serialize the model in json format
LSTM_MODEL = 'modellstm.json'

# and store the weights in h5
MODEL_WEIGHTS = 'modellstm.h5'


# Load the data and dump a short summary of the resulting DataFrame.

# In[5]:


train_df = pd.read_pickle(SHARE_ROOT + TRAIN_DATA)
train_df.head(10)


# In[6]:


test_df = pd.read_pickle(SHARE_ROOT + TEST_DATA)

test_df.head(10)


# ## Modelling
# 
# The traditional predictive maintenance machine learning models are based on feature engineering, the manual construction of variable using domain expertise and intuition. This usually makes these models hard to reuse as the feature are specific to the problem scenario and the available data may vary between customers. Perhaps the most attractive advantage of deep learning they automatically do feature engineering from the data, eliminating the need for the manual feature engineering step.
# 
# When using LSTMs in the time-series domain, one important parameter is the sequence length, the window to examine for failure signal. This may be viewed as picking a `window_size` (i.e. 5 cycles) for calculating the rolling features in the [Predictive Maintenance Template](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3). The rolling features included rolling mean and rolling standard deviation over the 5 cycles for each of the 21 sensor values. In deep learning, we allow the LSTMs to extract abstract features out of the sequence of sensor values within the window. The expectation is that patterns within these sensor values will be automatically encoded by the LSTM.
# 
# Another critical advantage of LSTMs is their ability to remember from long-term sequences (window sizes) which is hard to achieve by traditional feature engineering. Computing rolling averages over a window size of 50 cycles may lead to loss of information due to smoothing over such a long period. LSTMs are able to use larger window sizes and use all the information in the window as input. 
# 
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/ contains more information on the details of LSTM networks.
# 
# This notebook illustrates the LSTM approach to binary classification using a sequence_length of 50 cycles to predict the probability of engine failure within 30 days.

# In[7]:


# pick a large window size of 50 cycles
sequence_length = 50


# We use the [Keras LSTM](https://keras.io/layers/recurrent/) with [Tensorflow](https://tensorflow.org) as a backend. Here layers expect an input in the shape of an array of 3 dimensions (samples, time steps, features) where samples is the number of training sequences, time steps is the look back window or sequence length and features is the number of features of each sequence at each time step.
# 
# We define a function to generate this array, as we'll use it repeatedly.

# In[8]:


# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# The sequences are built from the features (sensor and settings) values across the time steps (cycles) within each engine. 

# In[9]:


# pick the feature columns 
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
key_cols = ['id', 'cycle']
label_cols = ['label1', 'label2', 'RUL']

input_features = test_df.columns.values.tolist()
sensor_cols = [x for x in input_features if x not in set(key_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(label_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(sequence_cols)]

# The time is sequenced along
# This may be a silly way to get these column names, but it's relatively clear
sequence_cols.extend(sensor_cols)

print(sequence_cols)


# In[10]:


# generator for the sequences
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
           for id in train_df['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
seq_array.shape


# We also create a function to label these sequences.

# In[11]:


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# We will only be using the LSTM to predict failure within the next 30 days (`label1`). To predict other labels, we could change this call before building the LSTM network.

# In[12]:


# generate labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape


# ## LSTM Network
# 
# Building a Neural Net requires determining the network architecture. In this scenario we will build a network of only 2 layers, with dropout. The first LSTM layer with 100 units, one for each input sequence, followed by another LSTM layer with 50 units. We will also apply dropout each LSTM layer to control overfitting. The final dense output layer employs a sigmoid activation corresponding to the binary classification requirement.

# In[13]:


# build the network
# Feature weights
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

# LSTM model
model = Sequential()

# The first layer
model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))

# Plus a 20% dropout rate
model.add(Dropout(0.2))

# The second layer
model.add(LSTM(
          units=50,
          return_sequences=False))

# Plus a 20% dropout rate
model.add(Dropout(0.2))

# Dense sigmoid layer
model.add(Dense(units=nb_out, activation='sigmoid'))

# With adam optimizer and a binary crossentropy loss. We will opimize for model accuracy.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Verify the architecture 
print(model.summary())


# It takes about 15 seconds per epoch to build this model on a DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu) using only CPU compute.

# In[14]:


#get_ipython().run_cell_magic('time', '', "# fit the network\nmodel.fit(seq_array, # Training features\n          label_array, # Training labels\n          epochs=10,   # We'll stop after 10 epochs\n          batch_size=200, # \n          validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)\n          verbose=1, #\n          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss\n                                                     min_delta=0,    # until it doesn't change (or gets worse)\n                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving\n                                                     verbose=0, \n                                                     mode='auto')]) ")


# We optimized the network weights on the training set accuracy, which we examine here. 

# In[16]:


# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Training Accurracy: {}'.format(scores[1]))
#run_logger.log("Training Accuracy", scores[1])


# We can examine the training set performance by looking at the model confusion matrix. Accurate predictions lie along the diagonal of the matrix, errors are on the off diagonal.

# In[17]:


# make predictions and compute confusion matrix
y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)
y_true = label_array
print('Training Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true, y_pred)
cm


# Since we have many more healthy cycles than failure cycles, we also look at precision and recall. In all cases, we assume the model threshold is at $Pr = 0.5$. In order to tune this, we need to look at a test data set. 

# In[18]:


# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)
print( 'Training Precision: ', precision, '\n', 'Training Recall: ', recall, '\n', 'Training F1 Score:', f1)
#run_logger.log("Training Precision", precision)
#run_logger.log("Training Recall", recall)
#run_logger.log("Training F1 Score", f1)


# ## Model testing
# Next, we look at the performance on the test data. Only the last cycle data for each engine id in the test data is kept for testing purposes. In order to compare the results to the template, we pick the last sequence for each id in the test data.

# In[19]:


seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
seq_array_test_last.shape


# We also ned the test set labels in the correct format.

# In[20]:


y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]

label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
label_array_test_last.shape

print(seq_array_test_last.shape)
print(label_array_test_last.shape)


# Now we can test the model with the test data. We report the model accuracy on the test set, and compare it to the training accuracy. By definition, the training accuracy should be optimistic since the model was optimized for those observations. The test set accuracy is more general, and simulates how the model was intended to be used to predict forward in time. This is the number we should use for reporting how the model performs.

# In[21]:


# test metrics
scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
print('Test Accurracy: {}'.format(scores_test[1]))
#run_logger.log("Test Accuracy", scores_test[1])


# Similarly for the test set confusion matrix. 

# In[22]:


# make predictions and compute confusion matrix
y_pred_test = model.predict_classes(seq_array_test_last)
y_true_test = label_array_test_last
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true_test, y_pred_test)
cm


# The confusion matrix uses absolute counts, so comparing the test and training set confusion matrices is difficult. Instead, it is  better to use precision and recall. 
# 
#  * _Precision_ measures how accurate your model predicts failures. What percentage of the failure predictions are actually failures.
#  * _Recall_ measures how well the model captures thos failures. What percentage of the true failures did your model capture.
#  
# These measures are tightly coupled, and you can typically only choose to maximize one of them (by manipulating the probability threshold) and have to accept the other as is.
# 

# In[23]:


# compute precision and recall
precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print( 'Test Precision: ', precision_test, '\n', 'Test Recall: ', recall_test, '\n', 'Test F1 Score:', f1_test)
#run_logger.log("Test Precision", precision_test)
#run_logger.log("Test Recall", recall_test)
#run_logger.log("Test F1 Score", f1_test)


# ## Saving the model  
# 
# The LSTM network is made up of two components, the architecture and the model weights. We'll save these model components in two files, the architecture in a `json` file that the `keras` package can use to rebuild the model, and the weights in an `HDF5` heirachy that rebuild the exact model. 

# In[24]:


# Save the model for operationalization: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import os
import h5py
from sklearn import datasets 
 
# save model
# serialize model to JSON
model_json = model.to_json()
with open(LSTM_MODEL, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(MODEL_WEIGHTS)
print("Model saved")


# To test the save operations, we can reload the model files into a test model `loaded_model` and rescore the test dataset.

# In[25]:


from keras.models import model_from_json

print(keras.__version__)

# load json and create model
json_file = open(LSTM_MODEL, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_WEIGHTS)

loaded_model.compile('sgd','mse')
print("Model loaded")


# The model constructed from storage can be used to predict the probability of engine failure.

# In[26]:


score = loaded_model.predict_proba(seq_array,verbose=1)
print(score.shape)
print(score)


# # Persist the model
# 
# In order to pass the model to our next notebook, we will write the model files to the shared folder within the Azure ML Workbench project. https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-read-write-files
# 
# In the `Code\3_operationalization.ipynb` Jupyter notebook, we will create the functions needed to operationalize and deploy any model to get realtime predictions. The artifacts created will be stored in one of your Azure storage containers for you to deploy and test your own web service.

# In[27]:


with open(SHARE_ROOT + LSTM_MODEL, 'wt') as json_file:
    json_file.write(model_json)
    print("json file written shared folder")
    json_file.close()
    
model.save_weights(os.path.join(SHARE_ROOT, MODEL_WEIGHTS))


# In[28]:


get_ipython().run_line_magic('ls', '')

