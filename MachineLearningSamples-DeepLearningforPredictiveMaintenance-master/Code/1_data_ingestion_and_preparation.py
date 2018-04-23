
# coding: utf-8

# # Step 1: Data Ingestion & Preparation
# 
# In this scenario, we build a LSTM network for the data set and scenario described at [Predictive Maintenance](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) to predict remaining useful life of aircraft engines. In summary, the scenario uses simulated aircraft values from 21 sensors to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.
# 
# The data ingestion notebook will download the simulated predicitive maintenance data sets from a public Azure Blob Storage. Labels are created from the `truth` data and joined to the `training` and `test` data. After some preliminary data cleaning and verification, the results are stored in a local (to the notebook server) folder for use in the remaining notebooks of this analysis.

# In[1]:


# import the libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import glob
import urllib

#from azureml.logging import get_azureml_logger
#run_logger = get_azureml_logger()
#run_logger.log('amlrealworld.predictivemaintenanceforpm.dataingestionpreparation','true')


# ## Download simulated data sets
# We will be reusing the raw simulated data files from the [Predictive Maintenance](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) tutorial. The notebook automatically downloads these files stored at http://azuremlsamples.azureml.net/templatedata/
# 
# The three data files are:
# 
#     * `PM_train.txt`
#     * `PM_test.txt`
#     * `PM_truth.txt`
#     
# This notebook labels the train and test set and does some preliminary cleanup. We create some summary graphics for each data set to verify the data download, and store the resulting data sets in a local folder.

# In[2]:


# The raw train data is stored on Azure Blob here:
basedataurl = "http://azuremlsamples.azureml.net/templatedata/"

# We will store each of these data sets in a local persistance folder
SHARE_ROOT = '../../Data/'

# These file names detail where we store each data file. 
TRAIN_DATA = 'PM_train_files.pkl'
TEST_DATA = 'PM_test_files.pkl'
TRUTH_DATA = 'PM_truth_files.pkl'


# ## Data Ingestion
# In the following section, we ingest the training, test and ground truth datasets from azure storage. The training data consists of multiple multivariate time series with `cycle` as the time unit, together with 21 sensor readings and 3 settings for each cycle. Each time series can be assumed as being generated from a different engine of the same type. The testing data has the same data schema as the training data. The only difference is that the data does not indicate when the failure occurs. Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data. You can find more details about the type of data used for this notebook at [Predictive Maintenance Template](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3).
# 
# The training data consists of data from 100 engines (`id`) in the form of multivariate time series with `cycle` as the unit of time with 21 sensor readings `s1:s21` and 3 operational `setting` features for each `cycle`. In this simulated data, an engine is assumed to be operating normally at the start of each time series. Ebgine degradation progresses and grows in magnitude until a predefined threshold is reached where the engine is considered unsafe for further operation. In this simulation, the last cycle in each time series can be considered as the failure point of the corresponding engine.

# In[3]:


# Load raw training data from Azure blob
train_df = 'PM_train.txt'

# Download the file once, and only once.
if not os.path.isfile(train_df):
    urllib.request.urlretrieve(basedataurl+train_df, train_df)

# read training data 
train_df = pd.read_csv('PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_df.head()


# The testing data has the same data schema as the training data except the failure point is unknown.

# In[4]:


# Load raw data from Azure blob
test_df = 'PM_test.txt'

# Download the file once, and only once.
if not os.path.isfile(test_df):
    urllib.request.urlretrieve(basedataurl+test_df, test_df)
    
# read test data
test_df = pd.read_csv('PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = train_df.columns


test_df.head()


# The ground truth data provides the number of remaining working cycles (Ramaining useful life (RUL)) for the engines in the testing data. We use this data to evaluation the model after training with the training data set only.

# In[5]:


# Load raw data from Azure blob
truth_df = 'PM_truth.txt'

# Download the file once, and only once.
if not os.path.isfile(truth_df):
    urllib.request.urlretrieve(basedataurl+truth_df, truth_df)
    
# read ground truth data
truth_df = pd.read_csv('PM_truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

truth_df.head()


# ## Data Preprocessing
# We next generate labels for the training data. Since the last observation is assumed to be a failure point, we can calculate the Remaining Useful Life (`RUL`) for every cycle in the data.

# In[6]:


# Data Labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)
train_df.head()


# Using RUL, we can create a label indicating time to failure. We define a boolean (`True\False`) value for `label1` indicating the engine will fail within 30 days (RUL $<= 30$). We can also define a multiclass `label2` $\in \{0, 1, 2\}$ indicating {Healthy, RUL <=30, RUL <=15} cycles. 

# In[7]:


# generate label columns for training data
w1 = 30
w0 = 15

# Label1 indicates a failure will occur within the next 30 cycles.
# 1 indicates failure, 0 indicates healthy 
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )

# label2 is multiclass, value 1 is identical to label1,
# value 2 indicates failure within 15 cycles
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2
train_df.head()


# In the [Predictive Maintenance Template](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) , cycle column is also used for training so we will also include the cycle column. Here, we normalize the columns in the training data.

# In[8]:


# MinMax normalization
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)
train_df.head()


# Next, we prepare the test data. We normalize the data using the same parameters from the training data normalization.

# In[9]:


test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)
test_df.head()


# Next, we use the ground truth dataset to generate labels for the test data.

# In[10]:


# generate column max for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

# generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)
test_df.head()


# We then create the same labels as used for the `training` data.

# In[11]:


# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
test_df.head()


# ## Data Visualization
# One critical advantage of LSTMs is their ability to remember from long-term sequences (window sizes) which is hard to achieve by traditional feature engineering as computing rolling averages over large window sizes (i.e. 50 cycles) may lead to loss of information due to smoothing and abstracting of values over such a long period. While feature engineering over large window sizes may not make sense, LSTMs are able to use all the information in the window as input.
# 
# We first look at an example of the sensor values for 50 cycles prior to the failure for engine `id = 3`. 

# In[12]:


# preparing data for visualizations 
# window of 50 cycles prior to a failure point for engine id 3
engine_id3 = test_df[test_df['id'] == 3]
engine_id3_50cycleWindow = engine_id3[engine_id3['RUL'] <= engine_id3['RUL'].min() + 50]
cols1 = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
engine_id3_50cycleWindow1 = engine_id3_50cycleWindow[cols1]
cols2 = ['s11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
engine_id3_50cycleWindow2 = engine_id3_50cycleWindow[cols2]

# plotting sensor data for engine ID 3 prior to a failure point - sensors 1-10 
ax1 = engine_id3_50cycleWindow1.plot(subplots=True, sharex=True, figsize=(20,20))


# In[13]:


# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax2 = engine_id3_50cycleWindow2.plot(subplots=True, sharex=True, figsize=(20,20))


# ## Persist the data sets
# 
# With the training and testing data created, we can turn our attention to modelling the engine failures. In order to pass the data set to out next notebook, we will write the data to a folder shared within the Azure ML Workbench project. https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-read-write-files
# 
# The `Code\2_model_building_and_evaluation.ipynb` Jupyter notebook will read these data files and train a LSTM network to predict the probability of engine failure within the next 30 cycles using the previous 50 cycles.

# In[17]:


# The data was read in using a Pandas data frame. We'll convert 
# store it for later manipulations in subsequent notebooks.
train_df.to_pickle(SHARE_ROOT + TRAIN_DATA)
test_df.to_pickle(SHARE_ROOT + TEST_DATA)

print("Data files saved!")

