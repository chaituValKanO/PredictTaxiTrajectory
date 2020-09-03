#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import csv
import copy
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec


# In[2]:


def get_clusters(coords):
    """
    Estimate clusters for the given list of coordinates.
    """
    # First, grossly reduce the spatial dataset by rounding up the coordinates to the 4th decimal
    # (i.e. 11 meters. See: https://en.wikipedia.org/wiki/Decimal_degrees)
    clusters = pd.DataFrame({
      'approx_latitudes': coords[:,0].round(4),
      'approx_longitudes': coords[:,1].round(4)
    })
    clusters = clusters.drop_duplicates(['approx_latitudes', 'approx_longitudes'])
    clusters = clusters.values

    # Further reduce the number of clusters
    # (Note: the quantile parameter was tuned to find a significant and reasonable number of clusters)
    bandwidth = estimate_bandwidth(clusters, quantile=0.0002)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(clusters)
    return ms.cluster_centers_


def tf_haversine(latlon1, latlon2):
    """
    Tensorflow version of the Haversine function to calculate distances between two sets of points.
    """
    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]

    REarth = 6371
    lat = tf.abs(lat1 - lat2) * np.pi / 180
    lon = tf.abs(lon1 - lon2) * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    a = tf.sin(lat / 2) * tf.sin(lat / 2) + tf.cos(lat1) * tf.cos(lat2) * tf.sin(lon / 2) * tf.sin(lon / 2)
    d = 2 * tf_atan2(tf.sqrt(a), tf.sqrt(1 - a))
    return REarth * d


def tf_atan2(y, x):
    """
    Tensorflow doesn't have an Atan2 function (at least not yet, see: https://github.com/tensorflow/tensorflow/issues/6095).
    So we define it here ourselves.
    """
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle



def equirectangular_distance(latlon1, latlon2):
    REarth = 6371
    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]
    eq_rect_dist = tf.sqrt(((lon2-lon1)*tf.cos((lat2-lat1)/2))**2 + (lat2-lat1)**2)
    return eq_rect_dist


# In[ ]:




