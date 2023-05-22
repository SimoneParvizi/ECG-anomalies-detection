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
#%%
from Helper import plot_time_series_single_class, train_model, predict, Encorder, Decoder, RAE

#%%
device = torch.device('cpu')



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

#%%
CLASS_NORMAL = 1.0
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


normal_df = final_df[final_df.targets == CLASS_NORMAL].drop(labels='targets', axis=1)
normal_df.shape
     

anomaly_df = final_df[final_df.targets != CLASS_NORMAL].drop(labels='targets', axis=1)
anomaly_df.shape
     

# %%
RANDOM_SEED=42
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33, 
  random_state=RANDOM_SEED
)
#%% Shuffling
train_df = train_df.sample(frac=1.0)
val_df = val_df.sample(frac=1.0)
#%%
train_df.shape
test_df.shape
val_df.shape
# %% Creating dataset


def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features
     


#%%


train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)
     


     


#%%
model = RAE(seq_len, n_features, 128)
model = model.to(device)

#%% Training


model, history = train_model(
    model,
    train_dataset,
    val_dataset,
    n_epochs=10
)

# %% Loss over training epochs
ax = plt.figure().gca()

ax.plot(history['train'])
ax.plot(history['val'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show()

#%%
# save model
torch.save(model, 'LSTM.pth')

#%%

_, losses = predict(model, train_dataset)

sns.distplot(losses, bins=50, kde=True)
#%% Evaluation on normal heartbeats from the test set

predictions, pred_losses = predict(model, test_normal_dataset)
sns.distplot(pred_losses, bins=50, kde=True)
     

Threshold = 25
correct = sum(loss <= Threshold for loss in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

#%% Evaluation on anomalies

anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

predictions, pred_losses = predict(model, anomaly_dataset)
sns.distplot(pred_losses, bins=50, kde=True)


correct = sum(loss <= Threshold for loss in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')
     

#%% Looking at the predictions vs the ground truth

def plot_prediction(data, model, title, ax):
  predictions, pred_losses = predict(model, [data])

  ax.plot(data, label='true')
  ax.plot(predictions[0], label='reconstructed')
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
  ax.legend()    





fig, axs = plt.subplots(
  nrows=2,
  ncols=6,
  sharey=True,
  sharex=True,
  figsize=(22, 8)
)

for i, data in enumerate(test_normal_dataset[:6]):
  plot_prediction(data, model, title='Normal', ax=axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
  plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

fig.tight_layout()