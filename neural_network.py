import time
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


import data_asset


def neural_network_predict(x_train, x_test, y_train, y_test):
    start_train_time = time.time()
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
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    start_test_time = time.time()
    predictions = model.predict(x_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    accuracy_train = model.evaluate(x_train, y_train, verbose=0)[1]*100
    accuracy_test = model.evaluate(x_test, y_test, verbose=0)[1]*100
    predict = predictions
    predict_len = len(predictions)

    # =============================
    print('\nThe accuracy of the Neural Network classifier on test data is = {:.2f}%'.format(
        accuracy_train))
    print('The accuracy of the Neural Network classifier on test data is = {:.2f}%'.format(
        accuracy_test))

    # Make predictions on test data
    print('Predictions: {}, {} data'.format(predictions, len(predictions)))

    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, predict, predict_len


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

t1 = time.time()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu,
                          input_shape=(7,)),  # input shape required
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

neural_network_predict(x_train, x_test, y_train, y_test)
