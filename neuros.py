import torch

X_train = torch.load("X_train.pt")
y_train = torch.load("y_train.pt")
X_val = torch.load("X_val.pt")
y_val = torch.load("y_val.pt")
W = torch.randn(11, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)