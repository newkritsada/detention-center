import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing


import dataAsset

print('\n\n===== Neural Network =====\n\n')
# print('dataAsset',dataAsset)

data = dataAsset.data_frame

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = pd.DataFrame(data, columns=dataAsset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)


# Reshape data for CNN
x_train = x_train.values.reshape(-1, len(dataAsset.feature), 1)
x_test = x_test.values.reshape(-1, len(dataAsset.feature), 1)

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

# Compile and fit model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2)

# Evaluate model on test data
accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
print('The accuracy on test data is {:.2f}%'.format(accuracy*100))

print('The accuracy of the Neural Network classifier on test data is = {:.2f}%'.format(
    accuracy*100))

# Make predictions on test data
predictions = model.predict(x_test)
print('Predictions: {}, {} data'.format(predictions, len(predictions)))


