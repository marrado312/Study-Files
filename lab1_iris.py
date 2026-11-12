"""Лабораторная работа 1. Учебная задача: ирисы Фишера."""
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# --- Формируем файл iris.data в формате UCI (без заголовка, CSV, 5 столбцов) ---
iris = load_iris()
names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
with open(os.path.join(OUT, "iris.data"), "w") as f:
    for row, t in zip(iris.data, iris.target):
        f.write(",".join(f"{v:.1f}" for v in row) + "," + names[t] + "\n")

# --- Пункт 2: загрузка genfromtxt, пятый столбец -> nan ---
data = np.genfromtxt(os.path.join(OUT, "iris.data"), delimiter=",")
print("=== Пункт 2: genfromtxt без dtype ===")
print(data[:5])

# --- Пункт 3 ---
print("\n=== Пункт 3: начальное исследование ===")
print("Data type :", type(data))
print("Data shape:", data.shape)
print(data[:10])

# --- Пункт 4: dtype=None ---
data1 = np.genfromtxt(os.path.join(OUT, "iris.data"), delimiter=",", dtype=None)
print("\n=== Пункт 4: dtype=None ===")
print(data1.shape)
print(type(data1))
print(type(data1[0]))
print(type(data1[0][4]))
print(data1[:10])

# --- Рис. 1.6: пользовательский dtype ---
dt = np.dtype("f8, f8, f8, f8, U30")
data2 = np.genfromtxt(os.path.join(OUT, "iris.data"), delimiter=",", dtype=dt)
print("\n=== Пользовательский dtype ===")
print(data2.shape)
print(type(data2))
print(type(data2[0]))
print(type(data2[0][4]))
print(data2[:10])

# --- Пункты 5-6: проекции ---
sl = [d[0] for d in data2]
sw = [d[1] for d in data2]
pl = [d[2] for d in data2]
pw = [d[3] for d in data2]
feats = {"Sepal Length": sl, "Sepal Width": sw, "Petal Length": pl, "Petal Width": pw}
styles = [("ro", "Setosa", slice(0, 50)), ("g^", "Versicolor", slice(50, 100)),
          ("bs", "Virginica", slice(100, 150))]


def projection(xn, yn, fname):
    plt.figure(figsize=(6, 4.5))
    for fmt, lab, s in styles:
        plt.plot(feats[xn][s], feats[yn][s], fmt, label=lab)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xn)
    plt.ylabel(yn)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close()


# Рис. 1.7 / 1.9 (три проекции из методички)
projection("Sepal Length", "Sepal Width", "iris_fig1_7.png")
projection("Sepal Length", "Petal Length", "iris_fig1_9b.png")
projection("Sepal Length", "Petal Width", "iris_fig1_9c.png")

# Все C(4,2)=6 проекций на одной фигуре
pairs = list(combinations(feats.keys(), 2))
print("\nЧисло проекций на плоскость (C(4,2)):", len(pairs))
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (xn, yn) in zip(axes.ravel(), pairs):
    for fmt, lab, s in styles:
        ax.plot(feats[xn][s], feats[yn][s], fmt, label=lab, markersize=4)
    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_title(f"{xn} - {yn}")
axes[0, 0].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "iris_all_projections.png"), dpi=110)
plt.close()
print("Готово")
