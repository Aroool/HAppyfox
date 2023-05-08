# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
     

data = pd.read_csv('D:/Edu_Exa_Scu/Dataset.csv')
data = data.drop(['Timestamp'] , axis = 1)
print("Sample Data",data.head(3))
print("Data Shape",data.shape)
print("Data Information",data.info())
print("Data Preprocessing")

new_data = data.dropna(axis = 0)
     

print("New Data Shape",new_data.shape)
     
print("Preprocessed Data Information",new_data.info())
     
print("Data Describe",new_data.describe())
ques = new_data.columns
ques
     
df = new_data.copy()
     

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df = new_data.apply(LabelEncoder().fit_transform)
df
df.describe()

df.info()

correlation_matrix = df.corr()
correlation_matrix['How are you feeling right now?'].sort_values(ascending = False)
     

#target varibale and feature

y = df['How are you feeling right now?']
X = df.drop(['How are you feeling right now?',
                       'Gender', 'Age', 'Relationship status',
                       'Which year are you in?','Have you used any social media within the last 6 hours?'], axis=1)
     

#split data into train and test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)
     

X_train.shape

X_valid.shape

#fit model
model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)


     

print("",model)

# accuracy on training set

model.score(X_train,y_train)
     
#checking validation accuracy

model.score(X_valid,y_valid)

model.predict(X_valid)

#changing max depth of the tree

train_acc = []
valid_acc = []

for depth in range (1,10):
  model = DecisionTreeClassifier(max_depth= depth, random_state= 10)
  model.fit(X_train, y_train)
  train_acc.append(model.score(X_train,y_train))
  valid_acc.append(model.score(X_valid,y_valid))
     

dataframe = pd.DataFrame({'max_depth':range(1,10), 'train_accuracy':train_acc, 'Valid_accuracy':valid_acc})
dataframe.head()

import matplotlib.pyplot as plt
     


plt.plot(dataframe['max_depth'], dataframe['train_accuracy'], marker = 'o')
plt.plot(dataframe['max_depth'], dataframe['Valid_accuracy'], marker = 'x')

plt.xlabel('depth of the tree')
plt.ylabel('accuracy')
plt.legend(['train_accuracy'], ['Valid_accuracy'], loc = "lower right")
plt.show()
     

# tuning the existing model

model = DecisionTreeClassifier(max_depth = 8, max_leaf_nodes= 25, random_state = 10)
     

model.fit(X_train,y_train)

model.score(X_train,y_train)

model.score(X_valid,y_valid)

model.predict(X_valid)