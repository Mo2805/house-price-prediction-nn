import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')




df = pd.read_csv("housing.csv")
print(f"\n Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 2. Fill missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())



q1, q3 = df['median_house_value'].quantile(0.25), df['median_house_value'].quantile(0.75)
upper  = q3 + 1.5 * (q3 - q1)
before = len(df)
df = df[df['median_house_value'] <= upper].reset_index(drop=True)
print(f" Outliers removed: {before - len(df)} rows (Price > ${upper:,.0f})")



df['median_house_value'] = np.log1p(df['median_house_value'])

print(f"   Value range after log: {df['median_house_value'].min():.2f} → {df['median_house_value'].max():.2f}")


le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])


X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]


scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 8. Save clean data
X_scaled["Price"] = y.values
X_scaled.to_csv("clean_housing.csv", index=False)




print(f"  Final shape     : {X_scaled.shape[0]} × {X_scaled.shape[1]}")

print(f"  Saved to        : clean_housing.csv")

# 9. Convert to float32
X_arr = X_scaled.drop(columns=["Price"]).values.astype(np.float32)
y_arr = X_scaled["Price"].values.astype(np.float32)

# 10. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.2, random_state=42
)
print(f" Train: {len(X_train)} | Test: {len(X_test)}")

# 11. Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)


# 12. DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64, shuffle=False)

print(f" DataLoader Ready — Batch size: 64 | Train batches: {len(train_loader)}")
