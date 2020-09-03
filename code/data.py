#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[3]:


def convert_coordinates(string):
  """
  Loads list of coordinates from given string and swap out longitudes & latitudes.
  We do the swapping because the standard is to have latitude values first, but
  the original datasets provided in the competition have it backwards.
  """
  return [(lat, long) for (long, lat) in json.loads(string)]


# In[5]:


def random_truncate(coords):
  """
  Randomly truncate the end of the trip's polyline points to simulate partial trips.
  This is only intended to be used for our custom train/validation/test datasets
  and not for the final test dataset provided by the competition as that one is
  already partial.
  """

    # There's no need to truncate if there's not more than one item
    if len(coords) <= 1:
        return coords

    # Pick a random number of items to be removed from the list.
    # (We do "-1" to ensure we have at least one item left)
    n = np.random.randint(len(coords)-1)
    #   print(f"Removing last {n} items from the trajectory")

    if n > 0:
      # Return the list without its last n items
      return coords[:-n]
    else:
      # No truncation needed in this case
      return coords


# In[6]:


def encode_feature(feature, train, test):
  """
  Encode the labels for the given feature across both the train and test datasets.
  """
  encoder = LabelEncoder()
  train_values = train[feature].copy()
  test_values = test[feature].copy()
  # Replace missing values with 0's so we can later encode them
  train_values[np.isnan(train_values)] = 0
  test_values[np.isnan(test_values)] = 0
  # Fit the labels across all possible values in both datasets
  encoder.fit(pd.concat([train_values, test_values]))
  # Add new column to the datasets with encoded values
  train[feature + '_ENCODED'] = encoder.transform(train_values)
  test[feature + '_ENCODED'] = encoder.transform(test_values)
  return encoder


# In[7]:


def extract_features(df):
  """
  Extract some features from the original columns in the given dataset.
  """
  # Convert polyline values from strings to list objects
  df['POLYLINE'] = df['POLYLINE'].apply(convert_coordinates)
  # Extract start latitudes and longitudes
  df['START_LAT'] = df['POLYLINE'].apply(lambda x: x[0][0])
  df['START_LONG'] = df['POLYLINE'].apply(lambda x: x[0][1])
  # Extract quarter hour of day
  datetime_index = pd.DatetimeIndex(df['TIMESTAMP'])
  df['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15   
  # Extract day of week
  df['DAY_OF_WEEK'] = datetime_index.dayofweek
  # Extract week of year
  df['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
  # Extract trip duration (GPS coordinates are recorded every 15 seconds)
  df['DURATION'] = df['POLYLINE'].apply(lambda x: 15 * len(x))


# In[8]:


def remove_outliers(df, labels):
    """
    Remove some outliers that could otherwise undermine the training's results.
    """
    # Remove trips that are either extremely long or short (potentially due to GPS recording issue)
    indices = np.where((df.DURATION > 60) & (df.DURATION <= 2 * 3600))
    df = df.iloc[indices]
    labels = labels[indices]
    
    # Remove trips that are too far away from Porto (also likely due to GPS issues)
    bounds = (  # Bounds retrieved using http://boundingbox.klokantech.com
        (41.052431, -8.727951),
        (41.257678, -8.456039)
    )
    indices = np.where(
        (labels[:,0]  >= bounds[0][0]) &
        (labels[:,1] >= bounds[0][1]) &
        (labels[:,0]  <= bounds[1][0]) &
        (labels[:,1] <= bounds[1][1])
    )
    df = df.iloc[indices]
    labels = labels[indices]
    
    return df, labels


# In[ ]:


def load_data():
  """
  Loads data from CSV files, processes and caches it in pickles for faster future loading.
  """

  datasets = []
  for kind in ['train', 'test']:
    # Load original CSV file
    csv_file = '../data/%s.csv' % kind
    df = pd.read_csv(csv_file)
    print(f"\nListing of first few rows in {kind} dataset \n")
    print(df.head(8))
    # Ignore items that are missing data
    df = df[df['MISSING_DATA'] == False]
    # Ignore items that don't have polylines
    df = df[df['POLYLINE'] != '[]']
    # Delete the now useless column to save a bit of memory
    df.drop('MISSING_DATA', axis=1, inplace=True)
    # Delete an apparently useless column (all values are 'A')
    df.drop('DAY_TYPE', axis=1, inplace=True)
    # Fix format of timestamps
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64[s]')
    # Extra some new features
    extract_features(df)
    datasets.append(df)
  return datasets

