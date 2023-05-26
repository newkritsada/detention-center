import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def knn_predict(x_train, x_test, y_train, y_test, feature_name):
    print('\n===== k-Nearest Neighbors =====\n')

    y_train = np.ravel(y_train)
    knn = KNeighborsClassifier(n_neighbors=3)

    start_train_time = time.time()
    knn.fit(x_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    start_test_time = time.time()
    predictions = knn.predict(x_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    accuracy_train = knn.score(x_train, y_train)*100
    accuracy_test = knn.score(x_test, y_test)*100
    precision = precision_score(y_test, predictions)*100
    recall = recall_score(y_test, predictions)*100
    predict = predictions
    predict_len = len(predictions)

    y_scores = knn.predict_proba(x_test)[:, 1]
    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # =============================
    if len(feature_name) > 0:
        print('\nFeature selection : {}'.format(feature_name))

    print('The accuracy training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(
        accuracy_test))
    print('The precision on test data is {:.2f}%'.format(
        precision))
    print('The recall on test data is {:.2f}%'.format(
        recall))

    # print('Predictions: {}, {} data'.format(predict, predict_len))
    print("\nClassification report\n",
          classification_report(y_test, predictions))

    # print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, precision, recall, predict, predict_len, training_time, testing_time


def dicision_tree_predict(x_train, x_test, y_train, y_test, feature_name):
    print('\n===== Dicision Tree =====\n')

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
    precision = precision_score(y_test, predictions)*100
    recall = recall_score(y_test, predictions)*100
    predict = predictions
    predict_len = len(predictions)

    # =============================
    if len(feature_name) > 0:
        print('\nFeature selection : {}'.format(feature_name))

    print('The accuracy on training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(accuracy_test))
    print('The precision on test data is {:.2f}%'.format(
        precision))
    print('The recall on test data is {:.2f}%'.format(
        recall))

    print("\nClassification report\n",
          classification_report(y_test, predictions))

    # print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, precision, recall, predict, predict_len, training_time, testing_time


def neural_network_predict(x_train, x_test, y_train, y_test, shape, feature_name):

    # # Reshape data for CNN
    x_train = x_train.values.reshape(-1, shape, 1)
    x_test = x_test.values.reshape(-1, shape, 1)

    # # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_reshape = tf.keras.utils.to_categorical(y_test, num_classes=2)

    start_train_time = time.time()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(shape,)),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    # Compile and fit model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])
    # , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()

    model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2)

    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    start_test_time = time.time()
    # Make predictions on the test data
    predictions = model.predict(x_test)

    new_predictions = []
    for predict in predictions:
        if predict[0] > predict[1]:
            new_predictions.append(0)
        else:
            new_predictions.append(1)

    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    accuracy_train = model.evaluate(x_train, y_train, verbose=0)[1]*100
    accuracy_test = model.evaluate(x_test, y_test_reshape, verbose=0)[1]*100
    precision = precision_score(y_test, new_predictions)*100
    recall = recall_score(y_test, new_predictions)*100

    predict = predictions
    predict_len = len(predictions)

    print('\n===== Neural Network =====\n')
    # =============================
    if len(feature_name) > 0:
        print('\nFeature selection : {}'.format(feature_name))

    print('\nThe accuracy of the Neural Network classifier on train data is = {:.2f}%'.format(
        accuracy_train))
    print('The accuracy of the Neural Network classifier on test data is = {:.2f}%'.format(
        accuracy_test))
    print('The precision on test data is {:.2f}%'.format(
        precision))
    print('The recall on test data is {:.2f}%'.format(
        recall))

    # Make predictions on test data
    # print('Predictions: {}, {} data'.format(predictions, len(predictions)))
    print("\nClassification report\n",
          classification_report(y_test, new_predictions))

    # print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, precision, recall, predict, predict_len, training_time, testing_time
