import numpy as np 
seedNr = 7
np.random.seed(seedNr)

import datetime
import time

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pandas_datareader.data as web

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler

from IPython.display import display, HTML
import cryptocompare
from talib import abstract

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import nltk
#nltk.download()
#nltk.download('averaged_perceptron_tagger')


#Neuronal Network
import tensorflow as tf

np.random.seed(seedNr)
tf.random.set_seed(seedNr)
from tensorflow import keras
