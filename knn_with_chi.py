from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

import data_asset
from knn import knn_predict


print('\n\n===== k-Nearest Neighbors =====\n\n')

data = data_asset.data_frame

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = data.drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
yy = data['ครั้งที่กระทำความผิด']
# XX = (XX - XX.mean()) / XX.std()

# Perform feature selection using chi-square test
X_new = data_asset.Chi(XX, yy)

x_train, x_test, y_train, y_test = train_test_split(
    X_new, yy, test_size=0.3, random_state=45)

accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time = knn_predict(
    x_train, x_test, y_train, y_test)

print(classification_report(y_test, predict))
print(confusion_matrix(y_test, predict))
