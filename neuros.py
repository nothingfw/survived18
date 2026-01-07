import torch
import csv

X_train = torch.load("X_train.pt")
y_train = torch.load("y_train.pt")
X_val = torch.load("X_val.pt")
y_val = torch.load("y_val.pt")

data = torch.load("newTest.pt")
id = torch.load("apartment_id.pt")

W = torch.randn(28, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
alpha = 0.01

for _ in range(1000):
    y_hat = X_train @ W + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y_train)
    loss.backward()
    with torch.no_grad():
        W -= alpha * W.grad
        b -= alpha * b.grad
    W.grad = None
    b.grad = None

with torch.no_grad():
    y_hat = torch.sigmoid(X_val @ W + b)
score = (y_hat - y_val).abs().mean()

X_test = torch.load("newTest.pt")

with torch.no_grad():
    logits = X_test @ W + b
    preds = torch.sigmoid(logits).squeeze().cpu().numpy()

apartment_ids = []
with open("newTest.csv", newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        apartment_ids.append(row[0])

assert len(apartment_ids) == len(preds)

with open("submission.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["apartment_id", "survived_to_18jan"])
    for apt_id, pred in zip(apartment_ids, preds):
        writer.writerow([apt_id, float(pred)])