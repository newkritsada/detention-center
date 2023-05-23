import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


import data_asset
from model import neural_network_predict


print('\n\n===== Neural Network =====\n\n')
# print('dataAsset',dataAsset)

data = data_asset.data_frame

XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)

neural_network_predict(x_train, x_test, y_train, y_test,len(data_asset.feature))
