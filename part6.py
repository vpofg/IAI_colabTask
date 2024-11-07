from sklearn import tree
import pandas as pd
import numpy as np
from IPython.display import Image
import pydotplus

df = pd.read_csv('DATASETS/swimming.csv')

clf_entropy = tree.DecisionTreeClassifier(criterion='entropy')
clf_gini = tree.DecisionTreeClassifier(criterion='gini')

print(df.head())

x = pd.get_dummies(df.drop('enjoy', axis=1))
y = df['enjoy']

clf = clf_entropy.fit(x, y)

pd.get_dummies(df.drop('enjoy', axis=1))

columns = pd.get_dummies(df.drop('enjoy', axis=1)).columns

dot_data = tree.export_graphviz(clf, out_file=None, rounded=True, filled=True, feature_names=columns)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())