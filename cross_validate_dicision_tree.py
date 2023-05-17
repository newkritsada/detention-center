import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from joblib import dump

from data_asset import feature, data_frame1, data_frame2, data_frame3, data_frame4, data_frame5, DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5


def dicision_tree_predict(x_train, x_test, y_train, y_test):
    # Create tree object
    decision_tree = tree.DecisionTreeClassifier(criterion='gini')

    # Train DT based on scaled training set
    start_train_time = time.time()
    decision_tree.fit(x_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    start_test_time = time.time()
    predictions = decision_tree.predict(x_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    accuracy_train = decision_tree.score(x_train, y_train)*100
    accuracy_test = decision_tree.score(x_test, y_test)*100
    predict = predictions
    predict_len = len(predictions)

    # Print performance
    print('The accuracy on training data is {:.2f}'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}'.format(accuracy_test))
    print('Predictions: {}, {} data'.format(predict, predict_len))
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, predict, predict_len


print('\n\n===== Dicision Tree (cross validate)=====\n\n')

data_trains=[DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
data_tests=[data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

results=[]
accuracy_train_sum=0
accuracy_test_sum=0
start_time=time.time()
for index, data_train in enumerate(data_trains):
    print("\nRound : ", index+1)
    x_train=pd.DataFrame(data_trains[index], columns=feature)
    y_train=pd.DataFrame(data_trains[index]['ครั้งที่กระทำความผิด'])

    x_test=pd.DataFrame(data_tests[index], columns=feature)
    y_test=pd.DataFrame(data_tests[index]['ครั้งที่กระทำความผิด'])
    accuracy_train, accuracy_test, predict, predict_len=dicision_tree_predict(
        x_train, x_test, y_train, y_test)

    accuracy_train_sum += accuracy_train
    accuracy_test_sum += accuracy_test

    results.append({
        'index': index,
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'predict': predict,
        'predict_len': predict_len
    })
end_time=time.time()
predict_all_result_time=end_time - start_time

print("\naccuracy train average is: {:.2f}%".format(
    accuracy_train_sum/len(data_tests)))
print("accuracy test average is: {:.2f}%".format(
    accuracy_test_sum/len(data_tests)))
print("Testing time:", predict_all_result_time)
