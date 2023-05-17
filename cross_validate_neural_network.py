import time
import pandas as pd
import tensorflow as tf


from data_asset import feature, data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5

from model import neural_network_predict


print('\n\n===== Neural Network (cross validate)=====\n\n')

data_trains = [DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
data_tests = [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

results = []
accuracy_train_sum = 0
accuracy_test_sum = 0
time_test_sum = 0

for index, data_train in enumerate(data_trains):
    print("\nRound : ", index+1)
    x_train = pd.DataFrame(data_trains[index], columns=feature)
    y_train = pd.DataFrame(data_trains[index]['ครั้งที่กระทำความผิด'])

    x_test = pd.DataFrame(data_tests[index], columns=feature)
    y_test = pd.DataFrame(data_tests[index]['ครั้งที่กระทำความผิด'])

    # Reshape data for CNN
    x_train = x_train.values.reshape(-1, len(feature), 1)
    x_test = x_test.values.reshape(-1, len(feature), 1)

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

    accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time = neural_network_predict(
        x_train, x_test, y_train, y_test)

    accuracy_train_sum += accuracy_train
    accuracy_test_sum += accuracy_test
    time_test_sum += testing_time

    results.append({
        'index': index,
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'predict': predict,
        'predict_len': predict_len
    })


print("\naccuracy train average is: {:.2f}%".format(
    accuracy_train_sum/len(data_tests)))
print("accuracy test average is: {:.2f}%".format(
    accuracy_test_sum/len(data_tests)))
print("summary Testing time:", time_test_sum)