# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:21:39 2020

@author: Lenovo
"""

from efficient_apriori import apriori

transactions = [('eggs','bacon','soup'),('eggs','bacon','apple'),('soup','bacon','banana')]
transactions

itemsets,rules = apriori(transactions)

print(itemsets)

print(rules)

x= filter(lambda x: len(x.lhs)==1 and len(x.rhs)==1,rules)
list(x)

for i in sorted(rules,key=lambda i: i.lift):
    print(i)


# 2 _AR 



transactions = [['milk', 'water'], ['milk', 'bread'], ['milk','bread','water']]
transactions

!pip install mlxtend

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary
te.columns_

import pandas as pd

df = pd.DataFrame(te_ary,columns=te.columns_)
df

from mlxtend.frequent_patterns import  apriori

freq_itemsets = apriori(df,use_colnames=(True))
freq_itemsets
type(freq_itemsets)

from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_columns',None)

confidence = association_rules(freq_itemsets,metric='confidence',min_threshold=0)
print(confidence)
print(confidence[['antecedents','consequents','support','confidence','lift']])

support = association_rules(freq_itemsets,metric='support',min_threshold=0)
print(support[['antecedents','consequents','support','confidence','lift']])

lift = association_rules(freq_itemsets,metric='lift',min_threshold=0)
print(lift[['antecedents','consequents','support','confidence','lift']])













# 3 _ AR 














df = pd.read_csv('store_data1.csv')
df
df.shape
df.columns

df.isnull().any()
df.iloc[0]
df.shape
df.values

'''
result = []
for i in range(0,7501):
    for j in range(0,20):
        if str(df.values[i][j])!='nan':
            result.append([str(df.values[i,j])])
'''
    

records = []

for i in range(0, 7501):
    print(i)
    records.append([str(df.values[i,j]) for j in range(0, 20) if str(df.values[i,j]) != 'nan'])

records

from efficient_apriori import apriori

itemsets,rules = apriori(records,min_support=0.001)    
itemsets   

rules
lst=[]
for i in sorted(rules,key=lambda i:i.lift):
    lst.append(i)

lst
dd= pd.DataFrame(lst)
dd.to_csv('test_Data.csv',index=False)
dd
# or 



from mlxtend.preprocessing import TransactionEncoder    
te= TransactionEncoder()
te_ar= te.fit(records).transform(records)
te_ar

df= pd.DataFrame(te_ar,columns=te.columns_)
df
pd.set_option('display.max_columns',None)

from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules

items = apriori(df,use_colnames=(True),min_support=0.001)
items

confidence = association_rules(items,metric='confidence',min_threshold=0)
confidence[['antecedents','consequents','support','confidence','lift']].head()

support = association_rules(items,metric='support',min_threshold=0)
support[['antecedents','consequents','support','confidence','lift']].head()

lift = association_rules(items,metric='lift',min_threshold=0)
lift[['antecedents','consequents','support','confidence','lift']].head()

lift[(lift.confidence>0.5) & (lift.lift>12)]


# 6_mbasket1 


import pandas as pd 
import numpy as np 
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('online_store.csv')
df.columns
df.head()

df['Description']

data = df.copy()

data.head()
data.isnull().any(axis=0)
data.dropna()
data
data.dropna(axis=0,inplace=True)
data.dropna(axis=0,inplace=True,subset=['Description'])
data

data['Description']=data['Description'].str.strip()
data['Description']

data.dtypes
data['InvoiceNo'].astype('str')
data.set_index('index')
    
data = data[~data['InvoiceNo'].str.contains('C')]
data

basket = df[df['Country']=='United Kingdom'].groupby(['InvoiceNo','Description'])['Quantity']
basket = basket.sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket.head()

basket.to_csv('checkingbasket.csv')

basket.head()


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.head()


basket_sets.values
basket_sets.columns


basket_sets.drop('POSTAGE', inplace=True, axis=1)

frq_items = apriori(basket_sets,min_support=0.03,use_colnames=True)
frq_items
frq_items.to_csv('frq_items.csv')


rules= association_rules(frq_items,metric='lift',min_threshold=1)
rules[['antecedents','consequents','confidence','support','lift']]

rules[ (rules['lift'] >= 6) &  (rules['confidence'] >= 0.8) ]

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()




'''
from mlxtend.preprocessing import TransactionEncoder
te= TransactionEncoder()
te_ar = te.fit(basket_sets).transform(basket_sets)
te.columns_

df = pd.DataFrame(te_ar,columns=te.columns_)
df

apriori(df,min_support=0.03,use_colnames=(True))
'''