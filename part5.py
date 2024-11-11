import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('DATASETS/cpu_vendor.csv')
df1 = pd.read_csv('DATASETS/cpu.csv')

# Task 3
datatypeV = df['vendor'].dtype
datatypeC = df['class'].dtype

print("Data type of the 'vendor' column:", datatypeV)
print("Data type of the 'class' column:", datatypeC)

# Task 5
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_cpu = df.drop('class', axis=1).to_numpy()
Y_cpu = df['class'].to_numpy()
X_cpuV = df1.drop('class', axis=1).to_numpy()
Y_cpuV = df1['class'].to_numpy()

X_cpu_train, X_cpu_test, Y_cpu_train, Y_cpu_test = train_test_split(X_cpu, Y_cpu, test_size=0.2, random_state=42)
X_cpuV_train, X_cpuV_test, Y_cpuV_train, Y_cpuV_test = train_test_split(X_cpuV, Y_cpuV, test_size=0.2, random_state=42)

model_cpu = LinearRegression()
model_cpuV = LinearRegression()
model_cpu.fit(X_cpu_train, Y_cpuV_train)
model_cpuV.fit(X_cpuV_train, Y_cpuV_train)

cpu_score = model_cpu.score(X_cpu_test, Y_cpu_test)
cpuV_score = model_cpuV.score(X_cpuV_test, Y_cpuV_test)
print("CPU Model Score:", cpu_score)
print("CPU vendor Model Score:", cpuV_score)

# Task 6
print("CPU Model Coefficients:", model_cpu.coef_)
print("CPU Model Intercept:", model_cpu.intercept_)
print("CPU Vendor Model Coefficients:", model_cpuV.coef_)
print("CPU Vendor Model Intercept:", model_cpuV.intercept_)

