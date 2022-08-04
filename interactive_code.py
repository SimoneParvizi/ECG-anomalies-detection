#%%
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



rcParams['figure.figsize']

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
