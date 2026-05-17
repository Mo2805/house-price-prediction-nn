# 🏠 House Price Prediction using MLP — California Housing

A Multilayer Perceptron (MLP) neural network trained to predict house prices, built as part of the Neural Networks Course Project. The project includes two models: a baseline model and an improved model using **PyTorch** with advanced regularization techniques.

---

## 📌 Problem Description

Given a set of district features such as location, number of rooms, population, and proximity to the ocean, the model predicts the **median house value** of a district in dollars. This is a **regression** task solved using a fully-connected Artificial Neural Network (MLP).

---

## 📦 Dataset

**Kaggle — California Housing Prices**
🔗 https://www.kaggle.com/datasets/camnugent/california-housing-prices

| Property | Details |
|---|---|
| Total Samples | 20,640 districts |
| Features | 9 features |
| Target | median_house_value ($) |
| Price Range | $14,999 – $500,001 |

---

## 🔧 Preprocessing Steps

1. **Filled missing values** — 207 missing values in `total_bedrooms` filled with median
2. **Removed outliers** — removed 1,071 rows with Price > $482,412 using IQR
3. **Log Transform** — applied `log1p` on target to reduce skewness
4. **Label Encoding** — encoded `ocean_proximity` (5 unique values)
5. **Normalization** — applied `StandardScaler` on all features
6. **Train / Test Split** — 80% training (15,655) / 20% testing (3,914)
7. **PyTorch Tensors** — converted to `FloatTensor` + `DataLoader` (batch size 64)

**Final shape: 19,569 × 10**

---

## 🧠 Model 1 — Baseline (Simple MLP)

```
Input  →  9 features
Hidden →  128 neurons (Tanh)
Hidden →   64 neurons (Tanh)
Hidden →   32 neurons (Tanh)
Output →   1 neuron (price)
```

| Metric | Result |
|---|---|
| Train R² | 0.8671 |
| Test R²  | 0.7998 |
| Test RMSE | $45,890 |
| Overfitting (diff) | 0.0673 ❌ |

---

## 🚀 Model 2 — Improved (BN + Dropout + Augmentation)

```
Input  →  9 features
Hidden →  128 neurons + BatchNorm + Tanh + Dropout(30%)
Hidden →   64 neurons + BatchNorm + Tanh + Dropout(30%)
Hidden →   32 neurons + BatchNorm + Tanh
Output →   1 neuron (price)
```

**Enhancement Techniques:**
- ✅ **Batch Normalization** — stabilizes and speeds up training
- ✅ **Dropout (30%)** — prevents overfitting
- ✅ **Data Augmentation** — adds Gaussian noise to training batches

| Metric | Result |
|---|---|
| Train R² | 0.8066 |
| Test R²  | 0.7982 |
| Test RMSE | $46,390 |
| Overfitting (diff) | **0.0085 ✅** |

---

## 📊 Model Comparison

| Metric | Model 1 | Model 2 |
|---|---|---|
| Batch Normalization | ❌ | ✅ |
| Dropout | ❌ | ✅ 30% |
| Data Augmentation | ❌ | ✅ |
| Train R² | 0.8671 | 0.8066 |
| Test R² | 0.7998 | 0.7982 |
| RMSE | $45,890 | $46,390 |
| Overfitting | 0.0673 ❌ | **0.0085 ✅** |

**Conclusion:** Model 2 reduced overfitting from **0.0673 to 0.0085** using Batch Normalization, Dropout, and Data Augmentation.

---

## 🔬 Experiments (4 Experiments — Model 2 Based)

| Experiment | What Changed | Train R² | Test R² | RMSE | Diff |
|---|---|---|---|---|---|
| Exp1 — Baseline | — | 0.7918 | 0.7856 | $46,343 | 0.0062 ✅ |
| Exp2 — Changed Activation | ReLU → Tanh | 0.7932 | 0.7837 | $47,033 | 0.0095 ✅ |
| Exp3 — Changed Neurons | 256→128→64 | 0.8112 | 0.7994 | $45,245 | 0.0118 ✅ |
| **Exp4 — Changed Dropout** ⭐ | **0.3 → 0.1** | **0.8237** | **0.8083** | **$45,183** | **0.0154 ✅** |

**Best: Exp4** — Reducing Dropout from 0.3 to 0.1 gave the highest Test R² (0.8083) and lowest RMSE ($45,183).

---

## ▶️ How to Run

### 1. Install Requirements
```bash
pip install scikit-learn pandas numpy matplotlib torch
```

### 2. Download the Dataset
Download `housing.csv` from the Kaggle link above and place it in the project folder.

### 3. Run Steps in Order
```bash
python ca_step1_preprocessing.py   # Preprocessing + Tensors
python ca_model1.py                # Model 1 — Baseline
python ca_model2.py                # Model 2 — Improved
python ca_experiments.py           # 4 Experiments
python ca_visualization.py         # Plots
```

---

## 📁 Repository Structure

```
├── ca_step1_preprocessing.py    # Preprocessing + DataLoader
├── ca_model1.py                 # Model 1 — Simple MLP
├── ca_model2.py                 # Model 2 — BN + Dropout + Augmentation
├── ca_experiments.py            # 4 Experiments
├── ca_visualization.py          # Visualization
├── clean_housing.csv            # Preprocessed dataset
├── visualization_ca.png         # Results charts
└── README.md                    # This file
```

---

## 📈 Evaluation Metrics

| Metric | Description | Best Result |
|---|---|---|
| **R²** | How much price variation the model explains | 0.8083 (Exp4) |
| **RMSE** | Average prediction error in dollars | $45,183 (Exp4) |
| **Diff** | Gap between Train and Test R² | 0.0062 (Exp1) |

---

## ▶️ How to Run

### 1. Install Requirements
```bash
pip install scikit-learn pandas numpy matplotlib torch
```

### 2. Download the Dataset
Download `housing.csv` from the Kaggle link above and place it in the project folder.

### 3. Run Steps in Order
```bash
python ca_step1_preprocessing.py   # Preprocessing + Tensors
python ca_model1.py                # Model 1 — Baseline
python ca_model2.py                # Model 2 — Improved
python ca_experiments.py           # 4 Experiments
python ca_visualization.py         # Plots
```
