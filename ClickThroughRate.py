'''
Created by Mahdi Mashayekhi
Email : MahdiMashayekhi.ai@gmail.com
Github : https://github.com/MahdiMashayekhi-AI
Site : http://mahdimashayekhi.gigfa.com
YouTube : https://youtube.com/@MahdiMashayekhi
Twitter : https://twitter.com/Mashayekhi_AI
LinkedIn : https://www.linkedin.com/in/mahdimashayekhi/

'''
'''
In any advertising agency, it is very important to predict the most profitable users who are very likely to respond to targeted advertisements. In this article, I’ll walk you through how to train a model for the task of click-through rate prediction with Machine Learning using Python.

'''
# Import libraries
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv('advertising.csv')
print(data.head())

# Check null values
print(data.isnull().sum())
print(data.columns)

# Here will drop some unnecessary columns:
x = data.iloc[:, 0:7]
x = x.drop(['Ad Topic Line', 'City'], axis=1)
y = data.iloc[:, 9]

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=4)

# Logistic Regression Model:
Lr = LogisticRegression(C=0.01, random_state=0)
Lr.fit(x_train, y_train)

# Prdiction of Model
y_pred = Lr.predict(x_test)
print(y_pred)

# Prdiction of Regression Model probabilities
y_pred_proba = Lr.predict_proba(x_test)
print(y_pred_proba)

print(accuracy_score(y_test, y_pred))

print(f1_score(y_test, y_pred))
