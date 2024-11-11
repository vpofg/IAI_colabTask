from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
from IPython.display import Image, display
import pydotplus

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

def create_and_visualize_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1, filename='decision_tree.png'):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf = clf.fit(x, y)
    
    dot_data = tree.export_graphviz(clf, out_file=None, rounded=True, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(filename)
    
    image = Image(filename)
    display(image)

create_and_visualize_tree(max_depth=3, filename='decision_tree_depth3.png')
create_and_visualize_tree(min_samples_split=10, filename='decision_tree_min_samples_split10.png')
create_and_visualize_tree(min_samples_leaf=5, filename='decision_tree_min_samples_leaf5.png')
