import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import preprocessing


from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

import data_asset

print('\n\n===== Dicision Tree =====\n\n')

data = data_asset.data_frame

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)

# Create tree object
decision_tree = tree.DecisionTreeClassifier(criterion='gini')

# Train DT based on scaled training set
decision_tree.fit(x_train, y_train)

# Print performance
print('The accuracy on training data is {:.2f}'.format(
    decision_tree.score(x_train, y_train)*100))
print('The accuracy on test data is {:.2f}'.format(
    decision_tree.score(x_test, y_test)*100))

# generate tree map
# dot_data = StringIO()
# export_graphviz(decision_tree, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Image(graph.write_png('output.png'))
# Image(graph.create_png())
