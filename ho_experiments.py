import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation, dropout=0.3):
        super(MLP, self).__init__()
        act_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        layers = []
        in_size = input_size
        for h in hidden_layers:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), act_fn, nn.Dropout(p=dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def augment(X_batch, noise=0.01):
    return X_batch + torch.randn_like(X_batch) * noise

def train_model(model, train_loader, lr=0.001, epochs=200, patience=20):
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    best_loss  = float('inf')
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = augment(X_batch)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
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
                break
    return best_loss

exps = [
    ("Exp1 — Baseline | Tanh | 128→64→32 | dropout=0.3",
     dict(hidden_layers=(128,64,32), activation='tanh', lr=0.001, dropout=0.3)),
    ("Exp2 — Changed Activation | ReLU | 128→64→32 | dropout=0.3",
     dict(hidden_layers=(128,64,32), activation='relu', lr=0.001, dropout=0.3)),
    ("Exp3 — Changed Neurons | Tanh | 256→128→64 | dropout=0.3",
     dict(hidden_layers=(256,128,64), activation='tanh', lr=0.001, dropout=0.3)),
    ("Exp4 — Changed Dropout | Tanh | 128→64→32 | dropout=0.1",
     dict(hidden_layers=(128,64,32), activation='tanh', lr=0.001, dropout=0.1)),
]

colors     = ['#2196F3', '#FF5722', "#2F8932", '#9C27B0']
results    = []
input_size = X_train_t.shape[1]



for name, params in exps:
    model      = MLP(input_size, params['hidden_layers'], params['activation'], params['dropout'])
    train_model(model, train_loader, lr=params['lr'])
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t).numpy().flatten()
        y_pred_test  = model(X_test_t).numpy().flatten()
    r2_train = r2_score(y_train_t.numpy(), y_pred_train)
    r2_test  = r2_score(y_test_t.numpy(),  y_pred_test)
    rmse     = np.sqrt(mean_squared_error(np.expm1(y_test_t.numpy().flatten()), np.expm1(y_pred_test)))
    diff     = r2_train - r2_test
    results.append((name, model, r2_train, r2_test, rmse))
    print(f"\n {name}")
    print(f"   Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | diff={diff:.4f} | RMSE=${rmse:,.0f}")

best = max(results, key=lambda x: x[3])
print(f"\n Best: {best[0]}")
print(f"   Test R²={best[3]:.4f} | RMSE=${best[4]:,.0f}")

pickle.dump({"results": results, "y_test": y_test_t.numpy().flatten(), "colors": colors},
            open("results_ca.pkl", "wb"))
print("\n Saved → results_ca.pkl")
