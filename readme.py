readme_content = """
# 🏠 House Price Prediction using MLP — California Housing

A Multilayer Perceptron (MLP) neural network trained to predict house prices, built as part of the Neural Networks Course Project. The project includes two models: a baseline model and an improved model using **PyTorch** with advanced regularization techniques.

---

## Problem Description

Given a set of district features such as location, number of rooms, population, and proximity to the ocean, the model predicts the **median house value** of a district in dollars. This is a **regression** task solved using a fully-connected Artificial Neural Network (MLP).

---

## Dataset

**Kaggle — California Housing Prices**
https://www.kaggle.com/datasets/camnugent/california-housing-prices

| Property        | Details              |
|-----------------|----------------------|
| Total Samples   | 20,640 districts     |
| Features        | 9 features           |
| Target          | median_house_value   |
| Price Range     | $14,999 – $500,001   |

---

## Preprocessing Steps

1. Filled missing values — 207 in total_bedrooms filled with median
2. Removed outliers — 1,071 rows removed (Price > $482,412) using IQR
3. Log Transform — applied log1p on target to reduce skewness
4. Label Encoding — ocean_proximity encoded (5 unique values)
5. Normalization — StandardScaler applied on all features
6. Train/Test Split — 80% train (15,655) / 20% test (3,914)
7. PyTorch Tensors — FloatTensor + DataLoader (batch size 64)

Final shape: 19,569 x 10

---

## Model 1 — Baseline (Simple MLP)

Architecture:
  Input  ->  9 features
  Hidden -> 128 neurons (Tanh)
  Hidden ->  64 neurons (Tanh)
  Hidden ->  32 neurons (Tanh)
  Output ->   1 neuron (price)

Results:
  Train R2   : 0.8671
  Test R2    : 0.7998
  Test RMSE  : $45,890
  Overfitting: 0.0673

---

## Model 2 — Improved (BN + Dropout + Augmentation)

Architecture:
  Input  ->  9 features
  Hidden -> 128 neurons + BatchNorm + Tanh + Dropout(30%)
  Hidden ->  64 neurons + BatchNorm + Tanh + Dropout(30%)
  Hidden ->  32 neurons + BatchNorm + Tanh
  Output ->   1 neuron (price)

Enhancement Techniques:
  - Batch Normalization : stabilizes and speeds up training
  - Dropout (30%)       : prevents overfitting
  - Data Augmentation   : adds Gaussian noise to training batches

Results:
  Train R2   : 0.8066
  Test R2    : 0.7982
  Test RMSE  : $46,390
  Overfitting: 0.0085

---

## Model Comparison

| Metric             | Model 1       | Model 2       |
|--------------------|---------------|---------------|
| Batch Normalization| No            | Yes           |
| Dropout            | No            | Yes (30%)     |
| Data Augmentation  | No            | Yes           |
| Train R2           | 0.8671        | 0.8066        |
| Test R2            | 0.7998        | 0.7982        |
| RMSE               | $45,890       | $46,390       |
| Overfitting        | 0.0673        | 0.0085        |

Conclusion: Model 2 reduced overfitting from 0.0673 to 0.0085.

---

## Experiments (4 Experiments — Model 2 Based)

| Experiment          | What Changed    | Test R2 | RMSE    | Diff   |
|---------------------|-----------------|---------|---------|--------|
| Exp1 — Baseline     | Nothing         | 0.7856  | $46,343 | 0.0062 |
| Exp2 — Activation   | ReLU -> Tanh    | 0.7837  | $47,033 | 0.0095 |
| Exp3 — Neurons      | 256->128->64    | 0.7994  | $45,245 | 0.0118 |
| Exp4 — Dropout (*)  | 0.3 -> 0.1      | 0.8083  | $45,183 | 0.0154 |

(*) Best: Exp4 — highest Test R2 (0.8083) and lowest RMSE ($45,183)

---

## How to Run

1. Install Requirements:
   pip install scikit-learn pandas numpy matplotlib torch

2. Download housing.csv from Kaggle link above

3. Run in order:
   python ca_step1_preprocessing.py
   python ca_model1.py
   python ca_model2.py
   python ca_experiments.py
   python ca_visualization.py

---

## Repository Structure

  ca_step1_preprocessing.py  -> Preprocessing + DataLoader
  ca_model1.py               -> Model 1 Simple MLP
  ca_model2.py               -> Model 2 BN + Dropout + Augmentation
  ca_experiments.py          -> 4 Experiments
  ca_visualization.py        -> Visualization
  clean_housing.csv          -> Preprocessed dataset
  README.md                  -> This file

---

## Evaluation Metrics

| Metric | Description                              | Best Result    |
|--------|------------------------------------------|----------------|
| R2     | Price variation explained by the model   | 0.8083 (Exp4)  |
| RMSE   | Average prediction error in dollars      | $45,183 (Exp4) |
| Diff   | Gap between Train and Test R2            | 0.0062 (Exp1)  |
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("✅ README.md saved!")
print(readme_content)
