import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from joblib import dump

from dataAsset import feature, data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5

print('\n\n===== k-Nearest Neighbors (cross validate)=====\n\n')
data_train = DataExcept1()
data_test = data_frame1

x_train = pd.DataFrame(data_train, columns=feature)
y_train = pd.DataFrame(data_train['ครั้งที่กระทำความผิด'])

x_test = pd.DataFrame(data_test, columns=feature)
y_test = pd.DataFrame(data_test['ครั้งที่กระทำความผิด'])


def knn_predict(x_train, x_test, y_train, y_test):
    y_train = np.ravel(y_train)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    accuracy_train = knn.score(x_train, y_train)*100
    accuracy_test = knn.score(x_test, y_test)*100
    predict = predictions
    predict_len = len(predictions)

    # =============================
    print('The accuracy training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(
        accuracy_test))
    print('Predictions: {}, {} data'.format(predict, predict_len))

    return accuracy_train, accuracy_test, predict, predict_len


accuracy_train, accuracy_test, predict, predict_len = knn_predict(
    x_train, x_test, y_train, y_test)
