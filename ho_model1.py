import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class Model1(nn.Module):
    def __init__(self, input_size):
        super(Model1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train_t.shape[1]
model1     = Model1(input_size)
criterion  = nn.MSELoss()
optimizer  = optim.Adam(model1.parameters(), lr=0.001)


epochs     = 200
best_loss  = float('inf')
patience   = 20
no_improve = 0

print("\n Training Model 1...")

for epoch in range(epochs):
    model1.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model1(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss  = avg_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 50 == 0:
        print(f"   Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

print(f"   Final Loss : {best_loss:.4f}")

model1.eval()
with torch.no_grad():
    y_pred_train = model1(X_train_t).numpy().flatten()
    y_pred_test  = model1(X_test_t).numpy().flatten()

r2_train  = r2_score(y_train_t.numpy(), y_pred_train)
r2_test   = r2_score(y_test_t.numpy(),  y_pred_test)
rmse_test = np.sqrt(mean_squared_error(
    np.expm1(y_test_t.numpy().flatten()),
    np.expm1(y_pred_test)
))


print(f"   Train R²  : {r2_train:.4f}")
print(f"   Test R²   : {r2_test:.4f}")
print(f"   Diff      : {r2_train - r2_test:.4f}")
print(f"   Test RMSE : ${rmse_test:,.0f}")

