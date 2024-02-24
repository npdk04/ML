import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
#import plotly.offline as pyoff
import plotly.graph_objs as go 
#import plotly.figure_factory as ff

# avoid displaying warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from multiscorer import MultiScorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
import time

# Loading the data
df = pd.read_csv('D:/nckh/ML/Data RFM.csv')
df.head()
df_data = df.dropna()
df_data.Date = pd.to_datetime(df_data.Date)
df_data.head()

ctm_bhvr_dt = df_data[(df_data.Date < 
                       pd.Timestamp(2020,7,1)) & 
                       (df_data.Date >= pd.Timestamp(2020,12,31))].reset_index(drop=True)

ctm_next_quarter = df_data[(df_data.Date < 
                            pd.Timestamp(2020,12,31)) & 
                            (df_data.Date >= pd.Timestamp(2020,7,1))].reset_index(drop=True)

ctm_dt = pd.DataFrame({'CustomerKey': ctm_bhvr_dt['CustomerKey'].unique()})


ctm_dt.head()