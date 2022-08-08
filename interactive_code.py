#%%
from email import header
import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from scipy.io.arff import loadarff 



rcParams['figure.figsize'] = 12.0, 9.0

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
df_TRAIN = pd.read_csv(r"C:\Users\simon\Desktop\Projects DL\ECG anomalies detection\ECG5000\ECG5000_TRAIN.txt", header = None, delim_whitespace=True)
df_TRAIN.head()


df_TEST = pd.read_csv(r"C:\Users\simon\Desktop\Projects DL\ECG anomalies detection\ECG5000\ECG5000_TEST.txt", header = None, delim_whitespace=True)
df_TEST.head()
df_TEST.shape
# %% concatenating the whole dataframe and checking it has the right lenght (5000x141)
final_df = pd.concat([df_TRAIN, df_TEST])
final_df.shape

#Renaming the target column
mapping = {final_df.columns[0]:'target (1,2,3,4,5)'}
final_df = final_df.rename(columns=mapping)
final_df.head()
# %%
