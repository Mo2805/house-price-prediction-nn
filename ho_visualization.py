import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


# Load Results

data    = pickle.load(open("results_ca.pkl", "rb"))
results = data["results"]
y_test  = data["y_test"]
colors  = data["colors"]
best    = max(results, key=lambda x: x[3])

# Reconstruct X_test tensor
df = pd.read_csv("clean_housing.csv")
X  = df.drop(columns=["Price"]).values.astype(np.float32)
y  = df["Price"].values.astype(np.float32)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_t = torch.FloatTensor(X_test)

# Actual vs Predicted (best model)
best[1].eval()
with torch.no_grad():
    y_pred_best = np.expm1(best[1](X_test_t).numpy().flatten())
y_real = np.expm1(y_test)


# Visualization

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#F5F7FA')

# Row 1 — Experiment Results Cards
for i, (name, model, r2_train, r2_test, rmse) in enumerate(results):
    ax = axes[0, i] if i < 3 else axes[0, 2]
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    label = name.split("|")[0].strip()
    ax.text(0.5, 0.6, label, ha='center', va='center',
            transform=ax.transAxes, fontsize=9, fontweight='bold', color=colors[i])
    ax.text(0.5, 0.4, f"Test R²  = {r2_test:.4f}",
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.25, f"RMSE = ${rmse:,.0f}",
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.1, f"Diff = {r2_train - r2_test:.4f}",
            ha='center', va='center', transform=ax.transAxes, fontsize=9,
            color='green' if r2_train - r2_test < 0.03 else 'red')
    ax.set_title(f"{label}", fontsize=8, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

# R² bar chart
names = [r[0].split("|")[0].strip() for r in results]
axes[1,0].bar(names, [r[3] for r in results], color=colors, edgecolor='white')
for i, val in enumerate([r[3] for r in results]):
    axes[1,0].text(i, val + 0.005, f"{val:.4f}", ha='center', fontweight='bold', fontsize=8)
axes[1,0].set_title("Test R² Comparison", fontweight='bold')
axes[1,0].set_ylim(0, 1.1); axes[1,0].grid(axis='y', alpha=0.3)
axes[1,0].set_facecolor('white')
axes[1,0].tick_params(axis='x', rotation=15)

# RMSE bar chart
axes[1,1].bar(names, [r[4] for r in results], color=colors, edgecolor='white')
for i, val in enumerate([r[4] for r in results]):
    axes[1,1].text(i, val + 100, f"${val:,.0f}", ha='center', fontweight='bold', fontsize=8)
axes[1,1].set_title("RMSE Comparison ($)", fontweight='bold')
axes[1,1].grid(axis='y', alpha=0.3); axes[1,1].set_facecolor('white')
axes[1,1].tick_params(axis='x', rotation=15)

# Actual vs Predicted
axes[1,2].scatter(y_real, y_pred_best, alpha=0.4, s=15, color=colors[results.index(best)])
lim = [y_real.min(), y_real.max()]
axes[1,2].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect')
axes[1,2].set_title(f"Actual vs Predicted (Best)", fontweight='bold')
axes[1,2].set_xlabel("Actual ($)"); axes[1,2].set_ylabel("Predicted ($)")
axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)
axes[1,2].set_facecolor('white')

plt.suptitle("Experiments — California Housing (PyTorch)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("visualization_ca.png", dpi=150, bbox_inches='tight')
print(" Saved → visualization_ca.png")
