import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from joblib import dump

import dataAsset

print('\n\n===== k-Nearest Neighbors (cross validate)=====\n\n')

data_train = dataAsset.DataExcept1()
x_test = dataAsset.data_frame1

le = preprocessing.LabelEncoder()
for col in data_train.loc[1:]:
    data_train[col] = le.fit_transform(data_train[col])

XX = pd.DataFrame(data_train, columns=dataAsset.feature)
yy = pd.DataFrame(data_train['ครั้งที่กระทำความผิด'])


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

    return {
        accuracy_train,
        accuracy_test,
        predict,
        predict_len
    }
