import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def knn_predict(x_train, x_test, y_train, y_test):
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
    print('The accuracy training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(
        accuracy_test))
    print('The precision on test data is {:.2f}%'.format(
        precision))
    print('The recall on test data is {:.2f}%'.format(
        recall))

    print('Predictions: {}, {} data'.format(predict, predict_len))
    print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time


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
    print('The accuracy on training data is {:.2f}%'.format(accuracy_train))
    print('The accuracy on test data is {:.2f}%'.format(accuracy_test))
    print('Predictions: {}, {} data'.format(predict, predict_len))
    print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time


def neural_network_predict(x_train, x_test, y_train, y_test, shape):
    start_train_time = time.time()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation=tf.nn.relu,
                              input_shape=(shape,)),  # input shape required
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
    print('\n')
    print("Training time:", training_time)
    print("Testing time:", testing_time)

    return accuracy_train, accuracy_test, predict, predict_len, training_time, testing_time
