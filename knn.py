import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from joblib import dump

import dataAsset

print('\n\n===== k-Nearest Neighbors =====\n\n')

data = dataAsset.data_frame

XX = pd.DataFrame(data, columns=dataAsset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)


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


knn_predict(x_train, x_test, y_train, y_test)
