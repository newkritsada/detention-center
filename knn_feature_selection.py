import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# from joblib import dump

import data_asset
from model import knn_predict
from feature_selection import XX_features

# print('\n\n===== K-Nearest Neighbors =====\n\n')

data = data_asset.data_frame

# XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])

def knn(XX,yy):   
  x_train, x_test, y_train, y_test = train_test_split(XX, yy, test_size=0.3, random_state=45)

  accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time = knn_predict(x_train, x_test, y_train, y_test)

  print("\n",classification_report(y_test, predict))
  print(confusion_matrix(y_test, predict))


for feature in XX_features:
  print('\n\n===== K-Nearest Neighbors =====\n')
  print('\nFeature selection : {}'.format(feature['feature']))
  knn(feature['XX'],yy)