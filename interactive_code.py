#%%
from email import header
from sre_parse import fix_flags
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
mapping = {final_df.columns[0]:'targets'}
final_df = final_df.rename(columns=mapping)
final_df.head()


# %% Shuffling data
final_df = final_df.sample(frac=1.0)
final_df.head()

#%% Data Exploration

final_df.targets.value_counts().plot(kind='barh')
final_df.targets.value_counts()
#they are imbalanced indeed. WHAT ARE THE LIMITS OF THIS IMBALANCE? WHEN IS IT ACCEPTABLE?



# %% Let's take a look at the different classes
def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(path_deviation.index, under_line, over_line, alpha=0.125)

    ax.set_title(class_name)

#%%
classes = final_df.targets.unique()
class_name = ['Normal', 'R on T', 'PVC', 'SP', 'UB']

fig, axs = plt.subplots(
    nrows=1,
    ncols=len(classes),
    sharey=True,
    figsize=(14,2)
)


for i, cls in enumerate(classes):
    ax = axs[i]
    data = final_df[final_df.targets == cls].drop(labels='targets', axis=1).mean(axis=0).to_numpy()
    plot_time_series_class(data, class_name[i], ax, n_steps=10)



# %% DATA PRE - PROCESSING 
#targets == 1 is 'Normal'
df_only_normal = final_df[final_df.targets == 1].drop(labels='targets', axis=1)
df_only_normal.shape


# %%
df_all_anomalies = final_df[final_df.targets != 1].drop(labels='targets', axis=1)
df_all_anomalies.shape
# %%
RANDOM_SEED=42
train_df, val_df = train_test_split(df_only_normal, test_size=0.15, random_state=RANDOM_SEED)

val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=RANDOM_SEED)

#%%
train_df.shape
test_df.shape
val_df.shape
# %% Creating dataset




#%% Building LSTM Autoencoder  
class Encorder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encorder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1
            batch_first=True
        ) 

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1
            batch_first=True
        )

    def forward(self,x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))


    

class Dencorder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Dencorder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1
            batch_first=True
        ) 

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1
            batch_first=True
        )

    def forward(self,x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))





class RAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):  
        super(RAE, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encorder(seq_len, n_features, embedding_dim).to_device(device)
        self.decoder = Decorder(seq_len, embedding_dim, n_features).to_device(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
