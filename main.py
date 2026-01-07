import torch
import numpy as np
import csv

# ======================
# reproducibility
# ======================
torch.manual_seed(42)
np.random.seed(42)

# ======================
# categorical mappings
# ======================
MAP_WING = {"south": 0, "north": 1, "east": 2, "west": 3}
MAP_WINDOW = {"old": 0, "normal": 1, "new": 2}
MAP_HEATING = {"central": 0, "electric_heater": 1}
MAP_TREE = {"spruce": 0, "fir": 1, "pine": 2}
MAP_FORM = {"dense": 0, "normal": 1, "sparse": 2}
MAP_STAND = {"simple_stand": 0, "bucket": 1, "water_reservoir": 2, "unknown": 3}
MAP_TINSEL = {"low": 0, "medium": 1, "high": 2}

# ======================
# helper: encode row
# ======================
def encode_row(row):
    row = row.copy()
    row[2]  = MAP_WING.get(row[2], np.nan)
    row[7]  = MAP_WINDOW.get(row[7], np.nan)
    row[8]  = MAP_HEATING.get(row[8], np.nan)
    row[17] = MAP_TREE.get(row[17], np.nan)
    row[19] = MAP_FORM.get(row[19], np.nan)
    row[20] = MAP_STAND.get(row[20], np.nan)
    row[25] = MAP_TINSEL.get(row[25], np.nan)
    return row

# ======================
# read & encode train
# ======================
X_rows = []
y_rows = []

with open("train.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        row = encode_row(row)
        X_rows.append([float(x) if x != "" else np.nan for x in row[1:-1]])
        y_rows.append(float(row[-1]))

X = torch.tensor(X_rows, dtype=torch.float32)
y = torch.tensor(y_rows, dtype=torch.float32).unsqueeze(1)

# ======================
# shuffle + split
# ======================
idx = torch.randperm(X.shape[0])
X = X[idx]
y = y[idx]

X_train_raw, X_val_raw = X[:20000], X[20000:]
y_train, y_val = y[:20000], y[20000:]

# ======================
# NaN handling (train mean)
# ======================
nan_mask = torch.isnan(X_train_raw)
col_mean = torch.where(nan_mask, 0.0, X_train_raw).sum(0) / (~nan_mask).sum(0).clamp(min=1)

X_train = torch.where(nan_mask, col_mean, X_train_raw)
X_val = torch.where(torch.isnan(X_val_raw), col_mean, X_val_raw)

# ======================
# normalization (train stats)
# ======================
mean = X_train.mean(0)
std = X_train.std(0)
std = torch.where(std < 1e-6, torch.ones_like(std), std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

X_train = torch.nan_to_num(X_train)
X_val = torch.nan_to_num(X_val)

# ======================
# save train artifacts
# ======================
torch.save(X_train, "X_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(y_val, "y_val.pt")
torch.save(mean, "train_mean.pt")
torch.save(std, "train_std.pt")

# ======================
# read & encode test
# ======================
X_test_rows = []
apartment_ids = []

with open("test.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        apartment_ids.append(row[0])
        row = encode_row(row)
        X_test_rows.append([float(x) if x != "" else np.nan for x in row[1:]])

X_test = torch.tensor(X_test_rows, dtype=torch.float32)

# ======================
# apply train preprocessing to test
# ======================
nan_mask = torch.isnan(X_test)
X_test = torch.where(nan_mask, col_mean, X_test)
X_test = (X_test - mean) / std
X_test = torch.nan_to_num(X_test)

# ======================
# save test artifacts
# ======================
torch.save(X_test, "newTest.pt")
torch.save(apartment_ids, "apartment_id.pt")
