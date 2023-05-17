import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

import dataAsset

print('\n\n===== k-Nearest Neighbors =====\n\n')

data = dataAsset.data_frame

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = data.drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
yy = data['ครั้งที่กระทำความผิด']
# XX = (XX - XX.mean()) / XX.std()

# Perform feature selection using Recursive feature test
X_new = dataAsset.Mutual(XX,yy)


x_train, x_test, y_train, y_test = train_test_split(
    X_new, yy, test_size=0.3, random_state=45)

y_train = np.ravel(y_train)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
predictions_string = le.inverse_transform(predictions)


# =============================
print('The accuracy training data is {:.2f}%'.format(
    knn.score(x_train, y_train)*100))
print('The accuracy on test data is {:.2f}%'.format(
    knn.score(x_test, y_test)*100))
print('Predictions: {}, {} data'.format(predictions, len(predictions)))

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
