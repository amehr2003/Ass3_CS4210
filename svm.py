#-------------------------------------------------------------------------
# AUTHOR: ANITA MEHRAZARIN
# FILENAME: "svm.py"
# SPECIFICATION: 
# FOR: CS 4210- Assignment #3
# TIME SPENT: 2 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

HYPERPARAMETER = []
MAX_ACCURACY= -9999999.999
for c_element in c:
    for d in degree:
        for k in kernel:
           for shape in decision_function_shape :

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                clf = svm.SVC(C=c_element, degree=d, kernel=k, decision_function_shape=shape)

                #Fit SVM to the training data
                #--> add your Python code here
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here
                count=0
                for(x_testSample, y_testSample) in zip(X_test, y_test):
                    if clf.predict([x_testSample]) [0]==y_testSample:
                        count+=1
                ACCURACY= count/len(y_test)

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here

                if (MAX_ACCURACY<ACCURACY):
                    MAX_ACCURACY=ACCURACY

                    print('Highest SVM accuracy so far:{0}, Parameters:{1}, degree:{2}, kernel:{3}, decision_function_shape:{4}'.format(MAX_ACCURACY, c_element, d, k, shape))





