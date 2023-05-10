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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



PATH = '/Users/simon/Desktop/ACHIEVE YOUR FUCKING DREAMS/Projects DL/'
df_TRAIN = pd.read_csv(f"{PATH}ECG anomalies detection/ECG5000/ECG5000_TRAIN.txt",
                        header = None, delim_whitespace=True)
df_TRAIN.head()
df_TRAIN.shape

df_TEST = pd.read_csv(f"{PATH}ECG anomalies detection/ECG5000/ECG5000_TEST.txt",
                        header = None, delim_whitespace=True)
df_TEST.head()
df_TEST.shape


final_df = pd.concat([df_TRAIN, df_TEST])
final_df.shape  #(5000x141)

# Renaming the target column
mapping = {final_df.columns[0]:'targets'}
final_df = final_df.rename(columns=mapping)
final_df.head()


# %% Shuffling data 
# PRIMA DI SHUFFLARE I DATI DOVRESTI PIGLIARTI L'HELDOUT SET
# final_df = final_df.sample(frac=1.0)
# final_df.head()

#%% EDA and Data Visualization

final_df.targets.value_counts().plot(kind='barh')
final_df.targets.value_counts()


# %% Let's take a look at the different classes
def plot_time_series_single_class(data, class_name, ax, n_steps):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(path_deviation.index, under_line, over_line, alpha=0.4)

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


for i, class_ in enumerate(classes):
    ax = axs[i]
    avaraged_data = final_df[final_df.targets == class_].drop(labels='targets', axis=1).mean(axis=0).to_numpy()
    plot_time_series_single_class(avaraged_data, class_name[i], ax, n_steps=10)


# %% DATA PRE - PROCESSING 
#targets == 1 is 'Normal'
df_only_normal = final_df[final_df.targets == 1].drop(labels='targets', axis=1)
df_only_normal.shape


# %%
df_all_anomalies = final_df[final_df.targets != 1].drop(labels='targets', axis=1)
df_all_anomalies.shape
# %%
RANDOM_SEED=42
train_df, validation_df = train_test_split(df_only_normal, test_size=0.3, random_state=RANDOM_SEED)
validation_df, test_df = train_test_split(validation_df, test_size=0.5, random_state=RANDOM_SEED)

#%% Shuffling
train_df = train_df.sample(frac=1.0)
validation_df = validation_df.sample(frac=1.0)
#%%
train_df.shape
test_df.shape
validation_df.shape
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


    

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Dencorder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2 * input_dim, output_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1
            batch_first=True
        ) 

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1
            batch_first=True
        )


        self.dense_layer = nn.Linear(self.hidden_dim, output_dim)

    def forward(self,x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.dense_layer(x)





class RAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):  
        super(RAE, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encorder(seq_len, n_features, embedding_dim).to_device(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to_device(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x



model = RAE(seq_len, n_features, embedding_dim=128)


#%% Training

def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []

        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())


        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f'Epoch {epoch}: train loss {train_loss}, val loss {val_loss}')

    
    return model.eval(), history




# %%
