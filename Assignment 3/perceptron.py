#-------------------------------------------------------------------------
# AUTHOR: Zihao Luo
# FILENAME: perceptron.py
# SPECIFICATION: This program answers the question of Complete the Python program (perceptron.py) and train a Single Layer Perceptron and a  Multi-Layer Perceptron to classify optically recognized handwritten digits. The program will use 3,823 training samples (optdigits.tra file) and 1,797 test samples (optdigits.tes file). Read the file  optdigits.names to get detailed information about this dataset, as well as the files optdigits-orig.tra and  optdigits-orig.names to see the original format of the data, and how it was transformed to speed-up the  learning process. You will compare the model performances while varying two hyperparameters,  learning rate and shuffle, checking which combination leads to the best predictions. Update and print  the accuracy of each classifier when it gets higher, together with their hyperparameters.

# FOR: CS 4210- Assignment #3
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('Assignment 3/optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

x_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('Assignment 3/optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

x_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highestPerceptronAcc = 0
highestMlpAcc = 0

for learningRate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for alg in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            
            if alg == 'Perceptron':
                clf = Perceptron(eta0=learningRate, shuffle=shuffle, max_iter=1000) 
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learningRate, hidden_layer_sizes=(25,), shuffle=shuffle, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(x_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            correctPrediction = 0
            for (x_testSample, y_testSample) in zip(x_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    correctPrediction += 1

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            accuracy = correctPrediction / len(y_test)
            
            if alg == 'Perceptron' and accuracy > highestPerceptronAcc:
                highestPerceptronAcc = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy:.2f}, Parameters: learning rate={learningRate}, shuffle={shuffle}")
            
            if alg == 'MLP' and accuracy > highestMlpAcc:
                highestMlpAcc = accuracy
                print(f"Highest MLP accuracy so far: {accuracy:.2f}, Parameters: learning rate={learningRate}, shuffle={shuffle}")
            
            











