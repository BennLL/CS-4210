#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: naive_bayes.py
# SPECIFICATION: The program wil answer the questionComplete the Python program (naïve_bayes.py) that will read the file  weather_training.csv (training set) and output the classification of each of the 10 instances from  the file weather_test (test set) if the classification confidence is >= 0.75. Sample of output: 
#               Day Outlook Temperature Humidity Wind PlayTennis Confidence 
#               D1003 Sunny Cool High Weak No 0.86 
#               D1005 Overcast Mild High Weak Yes 0.78 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Weak": 1, "Strong": 2}
play = {"Yes": 1, "No": 0}

X = []
Y = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])
            
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
            Y.append(play[row[5]])
#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
data = []
labels = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) 
    for row in reader:
        data.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])
        labels.append(row[0])


#Printing the header os the solution
#--> add your Python code here
print("Day PlayTennis Confidence")

# Making predictions
probabilities = clf.predict_proba(data)
predictions = clf.predict(data)

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for i in range(len(data)):
    confidence = max(probabilities[i])
    if confidence >= 0.75:
        play_tennis = "Yes" if predictions[i] == 1 else "No"
        print(f"{labels[i]} {play_tennis} {confidence}")

