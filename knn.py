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

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = pd.DataFrame(data, columns=dataAsset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


X_train, X_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)

# test_ids = X_test.index.tolist()
# X_test['CMST_CASE_JUVENILE_REF'] = test_ids

y_train = np.ravel(y_train)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# dump(knn, './model/knn_model.joblib')

# predictions_string = le.inverse_transform(predictions)
# print(predictions_string)
# results = pd.DataFrame({'CMST_CASE_JUVENILE_REF': test_ids, 'Prediction': predictions_string})
# print(results)

# =============================
print('The accuracy training data is {:.2f}%'.format(
    knn.score(X_train, y_train)*100))
print('The accuracy on test data is {:.2f}%'.format(
    knn.score(X_test, y_test)*100))
print('Predictions: {}, {} data'.format(predictions, len(predictions)))
