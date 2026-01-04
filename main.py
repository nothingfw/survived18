import torch
import numpy as np

file = open('train.csv', 'r')
text = file.read()

text = (text.replace('south', '0').replace('north', '1').replace('east', '2').replace('west', '3')
        .replace('new', '2').replace('old', '1').replace('normal', '0')
        .replace('electric_heater', '1').replace('central', '0')
        .replace('spruce', '0').replace('fir', '1').replace('pine', '2')
        .replace('dense', '0').replace('normal', '1').replace('sparse', '2')
        .replace('simple_stand', '0').replace('bucket', '1').replace('water_reservoir', '2').replace('unknown', '3')
        .replace('low', '0').replace('medium', '1').replace('high', '2'))
dataset = open("newTrain.csv", 'w')
dataset.write(text)
file.close()

data = np.genfromtxt('newTrain.csv', delimiter=',', skip_header=1)
data = torch.tensor(data, dtype=torch.float32)
data = data[:, 1:]
index = torch.randperm(data.shape[0])
data = data[index]

X, y = data[:, :-1], data[:, -1:]
X_train_raw, X_val_raw = X[:20_000], X[20_000:]
y_train, y_val = y[:20_000], y[20_000:]

train_nan = torch.isnan(X_train_raw)
val_nan = torch.isnan(X_val_raw)

col_sum = torch.where(train_nan, torch.zeros_like(X_train_raw), X_train_raw).sum(0)
col_cnt = (~train_nan).sum(0).clamp(min=1)
col_mean = col_sum / col_cnt

X_train = torch.where(train_nan, col_mean, X_train_raw)
X_val   = torch.where(val_nan,   col_mean, X_val_raw)

mean = X_train.mean(0)
std = X_train.std(0)
std = torch.where(std < 1e-6, torch.ones_like(std), std)

X_train = (X_train - mean) / std
X_val   = (X_val - mean) / std

torch.save(X_train, "X_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(y_val, "y_val.pt")