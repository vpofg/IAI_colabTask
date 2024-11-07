import pandas as pd
import numpy as np

df = pd.read_csv('swimming.csv')

length = len(df)

print(df.describe())

# Task 1
print(df)
print("The number of rows is: " + str(length)) 
print("The number of columns is: " + str(len(df.columns)))

# Task 2
columns = df.columns
print("Variables in the dataset:", list(columns))
feature_variables = list(columns.drop('enjoy'))
print("Feature variables:", feature_variables)

# Task 3
print("Total number of instances:", length)
positive_instances = len(df[df['enjoy'] == 'yes'])
negative_instances = len(df[df['enjoy'] == 'no'])
print("Positive instances (enjoy == yes):", positive_instances)
print("Negative instances (enjoy == no):", negative_instances)

# Task 4
# The answer would be temperature, as every time the temperature is warm, the 
# enjoy variable is always yes. And when the temperature is cool, the enjoy 
# variable is always no.

# Task 5
high_humidity_rows = df[df['humidity'] == 'high']
high_humidity_count = len(high_humidity_rows)
print("Number of items with humidity == 'high':", high_humidity_count)
high_humidity_indices = high_humidity_rows.index.tolist()
print("Indices of items with humidity == 'high':", high_humidity_indices)

