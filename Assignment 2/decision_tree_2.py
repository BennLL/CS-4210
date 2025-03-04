# -------------------------------------------------------------------------
# AUTHOR: Zihao Luo
# FILENAME: decision_tree_2.py
# SPECIFICATION: The program wil answer the question:Complete the Python program (decision_tree_2.py) that will read the files  contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv. Each training set has a different number of instances (10, 100, 1000 samples). You will observe that the trees  are being created by setting the parameter max_depth = 5, which is used to define the maximum depth  of the tree (pre-pruning strategy) in sklearn. Your goal is to train, test, and output the performance of  the 3 models created by using each training set on the test set provided (contact_lens_test.csv). You  must repeat this process 10 times (train and test using a different training set), choosing the average accuracy as the final classification performance of each model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv

feature_map = {
    "Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
    "Myope": 1, "Hypermetrope": 2,
    "No": 1, "Yes": 2,
    "Reduced": 1, "Normal": 2
}

class_map = {"Yes": 1, "No": 2}

dataSets = ['Assignment 2/contact_lens_training_1.csv',
            'Assignment 2/contact_lens_training_2.csv', 'Assignment 2/contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    for row in dbTraining:
        X.append([feature_map[row[0]], feature_map[row[1]], feature_map[row[2]], feature_map[row[3]]])

    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    for row in dbTraining:
        Y.append(class_map[row[4]])

    accuracy = 0
    # Loop your training and test tasks 10 times here
    for i in range(10):

        # Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        with open('Assignment 2/contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:
                    dbTest.append(row)
        
        correct = 0
        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            X_test = [feature_map[data[0]], feature_map[data[1]], feature_map[data[2]], feature_map[data[3]]]
            Y_test = class_map[data[4]]
            predicted = clf.predict([X_test])[0]

            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # --> add your Python code here
            if predicted == Y_test:
                correct += 1
        
        # Find the average of this model during the 10 runs (training and test set)
        # --> add your Python code here
        accuracy += correct / len(dbTest)
    
    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    print(f'final accuracy when training on {ds}: {accuracy / 10}')