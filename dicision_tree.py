import time
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


def dicision_tree_predict(x_train, x_test, y_train, y_test):
    # Create tree object
    decision_tree = tree.DecisionTreeClassifier(criterion='gini')

    # Train DT based on scaled training set
    start_train_time = time.time()
    decision_tree.fit(x_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    start_test_time = time.time()
    predictions = decision_tree.predict(x_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    accuracy_train = decision_tree.score(x_train, y_train)*100
    accuracy_test = decision_tree.score(x_test, y_test)*100
    predict = predictions
    predict_len = len(predictions)

    # Print performance
    print('The accuracy on training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(accuracy_test))
    print('Predictions: {}, {} data'.format(predict, predict_len))
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, predict, predict_len


print('\n\n===== Dicision Tree =====\n\n')

data = data_asset.data_frame

XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)

dicision_tree_predict(x_train, x_test, y_train, y_test)
