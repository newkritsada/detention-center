import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import data_asset
from model import neural_network_predict
from feature_selection import XX_features

# print('\n\n===== Neural Network =====\n\n')

data = data_asset.data_frame

# XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])

def neural_network(XX,yy,shape):
    XX = pd.DataFrame(XX)
    yy = pd.DataFrame(yy)
    x_train, x_test, y_train, y_test = train_test_split(
        XX, yy, test_size=0.3, random_state=45)

    
    accuracy_train, accuracy_test, precision, recall, predict, predict_len, training_time, testing_time = neural_network_predict(x_train, x_test, y_train, y_test, shape)

    # print("\n",classification_report(y_test, predict))
    # print(confusion_matrix(y_test, predict))


for feature in XX_features:
  print('\n\n===== Neural Network =====\n')
  print('\nFeature selection : {}'.format(feature['feature']))
  neural_network(feature['XX'],yy,len(feature['XX'][0]))
