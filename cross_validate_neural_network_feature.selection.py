import time
import pandas as pd
import tensorflow as tf


from data_asset import feature, data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5

from model import neural_network_predict
from feature_selection import feature_selections

print('\n\n===== Neural Network (cross validate)=====\n\n')

data_trains = [DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
data_tests = [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

def cross_validate_neural_network(data_trains,data_tests,feature_function):
    accuracy_train_sum = 0
    accuracy_test_sum = 0
    time_test_sum = 0

    for index, data_train in enumerate(data_trains):
        print("\nRound : ", index+1)
        x_train = data_trains[index].drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
        y_train = pd.DataFrame(data_trains[index]['ครั้งที่กระทำความผิด'])
        x_train_new = feature_function(x_train, y_train)
        x_train_new = pd.DataFrame(x_train_new)

        x_test = data_tests[index].drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
        y_test = pd.DataFrame(data_tests[index]['ครั้งที่กระทำความผิด'])
        x_test_new = feature_function(x_test, y_test)
        shape = len(x_test_new[0])
        x_test_new = pd.DataFrame(x_test_new)

        # Reshape data for CNN
        x_train_new = x_train_new.values.reshape(-1, shape, 1)
        x_test_new = x_test_new.values.reshape(-1, shape, 1)

        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
        
        accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time = neural_network_predict(
            x_train_new, x_test_new, y_train, y_test, shape)

        accuracy_train_sum += accuracy_train
        accuracy_test_sum += accuracy_test
        time_test_sum += testing_time


    print("\naccuracy train average is: {:.2f}%".format(
        accuracy_train_sum/len(data_tests)))
    print("accuracy test average is: {:.2f}%".format(
        accuracy_test_sum/len(data_tests)))
    print("summary Testing time:", time_test_sum)

for feature_selection in feature_selections:
    print('=======================================')
    print('\nFeature selection : {}'.format(feature_selection['feature_name']))
    cross_validate_neural_network(data_trains,data_tests, feature_selection['feature_function'])