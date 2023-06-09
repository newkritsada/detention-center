import time
import pandas as pd
import tensorflow as tf


from data_asset import feature, data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5

from model import neural_network_predict
from feature_selection import feature_selections

print('\n\n===== Neural Network (cross validate)=====\n\n')

data_frame_trains = [DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
data_frame_tests = [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

def cross_validate_neural_network(data_trains,data_tests,feature_name):
    accuracy_train_sum = 0
    accuracy_test_sum = 0
    precision_sum = 0
    recall_sum = 0
    time_test_sum = 0

    for index, data_train in enumerate(data_trains):
        print("\nRound : ", index+1)
        x_train = data_trains[index]
        y_train = pd.DataFrame(data_frame_trains[index]['ครั้งที่กระทำความผิด'])

        x_test = pd.DataFrame(data_tests[index], columns=data_trains[index].columns)
        y_test = pd.DataFrame(data_tests[index]['ครั้งที่กระทำความผิด'])

        shape = len(x_test.columns)

        # x_test_new = feature_function(x_test, y_test)

        
        accuracy_train, accuracy_test, precision, recall, predict, predict_len, training_time, testing_time = neural_network_predict(
            x_train, x_test, y_train, y_test, shape,feature_name)

        accuracy_train_sum += accuracy_train
        accuracy_test_sum += accuracy_test
        precision_sum += precision
        recall_sum += recall
        time_test_sum += testing_time

    print('\n===== Neural Network (cross validate)=====\n')

    print("\naccuracy train average is: {:.2f}%".format(
        accuracy_train_sum/len(data_tests)))
    print("accuracy test average is: {:.2f}%".format(
        accuracy_test_sum/len(data_tests)))
    print("precision test average is: {:.2f}%".format(
        precision_sum/len(data_tests)))
    print("recall test average is: {:.2f}%".format(
        recall_sum/len(data_tests)))
    print("summary Testing time:", time_test_sum)

for feature in feature_selections:
    cross_validate_neural_network(
        feature['data_trains'],
        feature['data_tests'],
        feature['feature_name']
        )