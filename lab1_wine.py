"""Лабораторная работа 1. Индивидуальное задание: набор данных Wine (UCI)."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["class"] = df["target"].map(dict(enumerate(wine.target_names)))
df.drop(columns="target", inplace=True)
df.to_csv(os.path.join(OUT, "wine.csv"), index=False)

print("=== 2.2 Форма набора ===")
print("Элементов:", df.shape[0], "| признаков:", df.shape[1] - 1, "| класс: 1 целевой")
print("\nТипы признаков:")
print(df.dtypes)
print("\nПропущенные значения по столбцам:")
print(df.isna().sum())
print("Всего пропусков:", int(df.isna().sum().sum()))
print("Дубликатов строк:", int(df.duplicated().sum()))
print("\nРаспределение классов:")
print(df["class"].value_counts())

feat = [c for c in df.columns if c != "class"]
desc = df[feat].describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["mean", "std", "min", "median", "max"]
print("\nОписательные статистики:")
print(desc.round(3))
desc.round(3).to_csv(os.path.join(OUT, "wine_stats.csv"))

print("\nСредние значения признаков по классам:")
print(df.groupby("class")[feat].mean().round(2).T)

# Выбросы по правилу 1.5*IQR
q1, q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
iqr = q3 - q1
outl = ((df[feat] < q1 - 1.5 * iqr) | (df[feat] > q3 + 1.5 * iqr)).sum()
print("\nЧисло выбросов (правило 1.5*IQR):")
print(outl[outl > 0])

# Корреляции
corr = df[feat].corr()
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
print("\nСамые сильные корреляции:")
print(pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).round(3))

# Графики
cls_colors = {"class_0": "tab:red", "class_1": "tab:green", "class_2": "tab:blue"}

# 1. гистограммы
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, c in zip(axes.ravel(), feat):
    ax.hist(df[c], bins=20, color="steelblue", edgecolor="white")
    ax.set_title(c, fontsize=9)
for ax in axes.ravel()[len(feat):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wine_hist.png"), dpi=100)
plt.close()

# 2. проекции (диаграммы рассеяния) по классам
projs = [("alcohol", "flavanoids"), ("alcohol", "color_intensity"),
         ("flavanoids", "proline"), ("total_phenols", "od280/od315_of_diluted_wines")]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (x, y) in zip(axes.ravel(), projs):
    for k, col in cls_colors.items():
        s = df[df["class"] == k]
        ax.scatter(s[x], s[y], c=col, label=k, s=18, alpha=.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
axes[0, 0].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wine_projections.png"), dpi=110)
plt.close()

# 3. boxplot по классам для нескольких признаков
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, c in zip(axes, ["alcohol", "flavanoids", "color_intensity", "proline"]):
    sns.boxplot(data=df, x="class", y=c, hue="class", palette=cls_colors, ax=ax, legend=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wine_box.png"), dpi=100)
plt.close()

# 4. матрица корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wine_corr.png"), dpi=100)
plt.close()

# 5. pairplot по 4 признакам
g = sns.pairplot(df[["alcohol", "flavanoids", "color_intensity", "proline", "class"]],
                 hue="class", palette=cls_colors, plot_kws={"s": 15}, height=2.2)
g.savefig(os.path.join(OUT, "wine_pairplot.png"), dpi=90)
print("\nГрафики сохранены")
