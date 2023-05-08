# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
     

df = pd.read_csv('D:/Edu_Exa_Scu/Dataset.csv')
print("",df.head(2))

df1 = df.iloc[:,1:].copy()
print(df1)
print(df1.isnull())

#preprocess_dataset

data = df1.to_numpy().tolist()
data
     
#convert all data to string type

for values in data:
    for i in range (len(values)):
        str1 = ''
        if type(values[i]) != str:
            str1 = str(values[i])
            values[i] = str1

tran_encoder = TransactionEncoder()

# encoding data to boolean values of array

data1 = tran_encoder.fit(data).transform(data)
print(data1)

new_df = pd.DataFrame(data1, columns = tran_encoder.columns_)
new_df
 

#creating frequent itemset using apriori

frequent_item = apriori(new_df, min_support= 0.5, use_colnames= True)
print(frequent_item)

#generating rule using association rule mining 

result = association_rules(frequent_item, metric= 'confidence', min_threshold= 0.8)
rules = result.loc[:, ['antecedents', 'consequents', 'support', 'lift', 'confidence']]
print(rules)

rules.describe()
     

z = rules['support'].min()
print('minimum support value:', z)

z1 = rules['support'].max()
print('maximum suport value:', z1)

print ('-------------------------------------------')

z2 = rules['confidence'].min()
print('minimum value of confidence:', z2)

z3 = rules['confidence'].max()
print('maximum value of confidence:', z3)

print ('-------------------------------------------')

index = rules.index
number_of_rows = len(index)

print('total number of rules being generated is:', number_of_rows)

sc = result.loc[:, ['support', 'confidence']]
     

import matplotlib.pyplot as plt

sc.plot.scatter(x = 'support', y = 'confidence', alpha = 0.5)
plt.title('support vs confidence - scatter plot')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('support vs lift - scatter plot')
plt.show()

fit = np.polyfit( rules['lift'],rules['confidence'] ,1)
fit1 = np.poly1d(fit)
plt.xlabel('lift')
plt.ylabel('confidence')

plt.plot(rules['lift'], rules['confidence'],'o', rules['lift'], fit1(rules['lift']))    