from sklearn import tree
import pandas as pd
import numpy as np
from IPython.display import Image, display
import pydotplus

df = pd.read_csv('DATASETS/swimming.csv')

clf_entropy = tree.DecisionTreeClassifier(criterion='entropy')
clf_gini = tree.DecisionTreeClassifier(criterion='gini')

print(df.head())

x = pd.get_dummies(df.drop('enjoy', axis=1))
y = df['enjoy']

clf = clf_entropy.fit(x, y)

columns = pd.get_dummies(df.drop('enjoy', axis=1)).columns

dot_data = tree.export_graphviz(clf, out_file=None, rounded=True, filled=True, feature_names=columns)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree.png')

image = Image('decision_tree.png')
display(image)

# Drop the 'airTemp' column
x2 = pd.get_dummies(df.drop(['enjoy', 'airTemp'], axis=1))
y2 = df['enjoy']

clf = clf_entropy.fit(x2, y2)

columns = x2.columns

dot_data = tree.export_graphviz(clf, out_file=None, rounded=True, filled=True, feature_names=columns)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree2.png')

image = Image('decision_tree2.png')
display(image)

# Predicting 'enjoy' 
instances = x2.head(5)
predictions = clf.predict(instances)
print("Predictions for the first 5 instances:", predictions)

new_instances = pd.DataFrame({
    'waterTemp': [75, 80],
    'humidity': [60, 70],
    'wind': [10, 5],
    'cloud': [20, 50]
})

new_instances = pd.get_dummies(new_instances).reindex(columns=columns, fill_value=0)
new_predictions = clf.predict(new_instances)
print("Predictions for new instances:", new_predictions)