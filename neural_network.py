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


# Reshape data for CNN
x_train = x_train.values.reshape(-1, len(data_asset.feature), 1)
x_test = x_test.values.reshape(-1, len(data_asset.feature), 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu,
                          input_shape=(7,)),  # input shape required
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

neural_network_predict(x_train, x_test, y_train, y_test)
