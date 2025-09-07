# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value. .

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: EZHILARASI N
RegisterNumber: 212224040088
*/

PROGRAM :
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

#categorizing col for further labelling
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

#display dependent variables
Y

#Inititalize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#Define the sigmoid function
def sigmoid(z):
    return 1 /(1 + np.exp(-z))

#Define the loss function
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

#Define the gradient descent algorithm

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha * gradient
    return theta

#Train the model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

#Make predictions
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

#Exaluate the model
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

### Dataset
<img width="1482" height="507" alt="image" src="https://github.com/user-attachments/assets/337943b0-7362-4b9a-9d03-09d2f90e3c80" />

### Dataset.dtypes

<img width="371" height="386" alt="image" src="https://github.com/user-attachments/assets/5779d171-92ad-4ca7-b5ed-c98d95bfdd4a" />

### Dataset
<img width="1175" height="562" alt="image" src="https://github.com/user-attachments/assets/44fe9ae6-303e-4a93-905d-bd291c0d893c" />
### Y
<img width="891" height="283" alt="image" src="https://github.com/user-attachments/assets/f5632cd0-2c42-46cf-818c-0fd3abea38bc" />
### Accuracy
<img width="422" height="56" alt="image" src="https://github.com/user-attachments/assets/6fbf1384-e748-463f-956d-ecf5f0d115d9" />
### Y_Pred
<img width="912" height="180" alt="image" src="https://github.com/user-attachments/assets/b6e49210-b49a-4287-a96a-10b0752b2303" />
### Y
<img width="921" height="178" alt="image" src="https://github.com/user-attachments/assets/f6078e62-20c8-44ea-bfc2-4e72c760094f" />
### Y_prednew
<img width="173" height="42" alt="image" src="https://github.com/user-attachments/assets/63edad46-4bbf-45e2-9f21-bf1a67f84b25" />
### Y_prednew
<img width="111" height="51" alt="image" src="https://github.com/user-attachments/assets/3e40614a-76cc-4e53-a4f4-1ad28cc75b7d" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

