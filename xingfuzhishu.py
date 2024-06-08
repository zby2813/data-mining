import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# 数据加载
df = pd.read_csv(r'C:\Users\zby\Downloads\archive (2)\better-life-index-2024.csv')

# 数据预处理
df.columns = df.columns.str.strip().str.replace(r'[\s()]+', '_', regex=True).str.lower()

# 数据可视化
plt.figure(figsize=(12, 12))
sns.heatmap(df.iloc[:,1:].corr(), cmap='coolwarm', annot=True, fmt='.1f')


plt.figure(figsize=(15,4))
sns.barplot(x='country', y='life_satisfaction', data=df.sort_values(by='life_satisfaction', ascending=False))
plt.xticks(rotation=90)
plt.title('Life Satisfaction by Country')
plt.show()

# 特征选择
X = df[['gdp_per_capita_usd_']]
y = df['life_satisfaction']

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 模型性能评估
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print("线性回归模型的均方误差:", mse.round(3))
print("线性回归模型的R-squared:", round(r_squared, 3))

# 岭回归模型选择和超参数调优
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)

# 输出最佳参数和最佳模型的性能
print("最佳参数:", ridge_regressor.best_params_)
print("最佳模型的负均方误差:", ridge_regressor.best_score_)

# 使用最佳模型进行预测
y_pred_ridge = ridge_regressor.predict(X_test)

# 计算新模型的均方误差和R平方值
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r_squared_ridge = r2_score(y_test, y_pred_ridge)
print("岭回归模型的均方误差:", mse_ridge.round(3))
print("岭回归模型的R-squared:", round(r_squared_ridge, 3))

