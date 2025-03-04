# -------------------------------------------------------------------------
# AUTHOR: Zihao Luo
# FILENAME: knn.py
# SPECIFICATION: The program wil answer the question:Complete the Python program (knn.py) to read the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task. The dataset consists of email samples, where each sample includes the counts of 20  specific words (e.g., “agenda” or “prize”) representing their frequency of occurrence.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

count = 0

#Loop your data to allow each instance to be your test set
for i in range (len(db)):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j in range(len(db)):
        if j != i:
            X.append([float(value) for value in db[j][:-1]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
            if db[j][-1] == 'spam':
                Y.append(1)
            else:
                Y.append(0)
    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    sample = [float(value) for value in db[i][:-1]]
    label = 0
    if db[i][-1] == 'spam':
        label = 1

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    predicted = clf.predict([sample])[0]
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if predicted != label:
        count += 1
    
    
#Print the error rate
#--> add your Python code here
print("Error Rate:", count/len(db))





