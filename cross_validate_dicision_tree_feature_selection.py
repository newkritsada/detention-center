import pandas as pd

# from joblib import dump

from data_asset import data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5

from model import dicision_tree_predict
from feature_selection import feature_selections

print('\n\n===== Dicision Tree (cross validate)=====\n\n')

data_trains = [DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
data_tests = [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

def cross_validate_dicision_tree(data_trains,data_tests,feature_function):
    accuracy_train_sum = 0
    accuracy_test_sum = 0
    time_test_sum = 0

    for index, data_train in enumerate(data_trains):
        print("\nRound : ", index+1)
        x_train = data_trains[index].drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
        y_train = pd.DataFrame(data_trains[index]['ครั้งที่กระทำความผิด'])
        x_train_new = feature_function(x_train, y_train)

        x_test = data_tests[index].drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
        y_test = pd.DataFrame(data_tests[index]['ครั้งที่กระทำความผิด'])
        x_test_new = feature_function(x_test, y_test)

        accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time = dicision_tree_predict(
            x_train, x_test, y_train, y_test)

        accuracy_train_sum += accuracy_train
        accuracy_test_sum += accuracy_test
        time_test_sum += testing_time


    print("\naccuracy train average is: {:.2f}%".format(
        accuracy_train_sum/len(data_tests)))
    print("accuracy test average is: {:.2f}%".format(
        accuracy_test_sum/len(data_tests)))
    print("Testing time:", time_test_sum)

for feature_selection in feature_selections:
    print('=======================================')
    print('\nFeature selection : {}'.format(feature_selection['feature_name']))
    cross_validate_dicision_tree(data_trains,data_tests, feature_selection['feature_function'])