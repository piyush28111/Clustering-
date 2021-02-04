# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:30:27 2020

@author: Lenovo
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 


data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'y': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
df= pd.DataFrame(data)
df

plt.scatter(df.x,df.y)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10,max_iter=300).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

kmeans.labels_

plt.figure(dpi=1000)
plt.scatter(df.x,df.y,c=kmeans.labels_,s=50)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',color='r')
plt.show()



sse=[]
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.plot(range(1,11),sse,marker='o')
plt.grid(True)
plt.xlabel('No. of clusters')
plt.ylabel('SSE')
plt.xticks(range(1,11))
plt.show()

kmeans= KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
centroids 
kmeans.labels_
plt.scatter(df.x,df.y,c=kmeans.labels_)
plt.scatter(centroids[::,0],centroids[::,1],marker='*',c='r',s=80)


!pip install kneed
from kneed import KneeLocator

KneeLocator(x=range(1,11), y=sse,curve='convex',direction='decreasing').elbow


kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
plt.figure(dpi=1000)
plt.scatter(df.x,df.y,c=kmeans.labels_)
plt.scatter(centroids[::,0],centroids[::,1],marker='*',s=80,c='r')
kmeans.inertia_



from pydataset import data
mtcars = data('mtcars')
mtcars.head()
df= mtcars.copy()
df
sse= []
kmeans_kwargs={'init':'random','n_init':10,'max_iter':300}
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,**kmeans_kwargs).fit(df)
    sse.append(kmeans.inertia_)
sse    
plt.plot(range(1,11),sse,marker='*')
plt.grid(True)
plt.xticks(range(1,11))

KneeLocator(x=range(1,11), y=sse,curve='convex',direction='decreasing').elbow

kmeans= KMeans(n_clusters=2)
kmeans.fit(df)
kmeans.cluster_centers_
kmeans.labels_

df['labels']=kmeans.labels_
df
#df.sort_values('labels',ascending=True)
pred = kmeans.predict(df.drop('labels',axis=1))
pred
df
df['predicted_label']=pred
df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit_transform(df)
scaler

kmeans = KMeans(n_clusters=2,init='random',n_init=10,max_iter=300).fit(scaler)
centroids=kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_
kmeans.n_iter_

df['labels1']=kmeans.labels_
df


## CT 02 


url='https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/data/clustering.csv'
data = pd.read_csv(url)
data

data.shape
data.head()
data.describe()
data.dtypes
data.columns
plt.scatter(data['ApplicantIncome'],data['LoanAmount'])
data.LoanAmount
data.ApplicantIncome


data.dtypes
data.isnull().any()
data.isnull().any(axis=1)
data.index[data.isnull().any(axis=1)]
#data.index[data['LoanAmount'].isnull()]
data.iloc[6]

data.isnull().sum().sum()
data.isnull().sum(axis=1)
data.isnull().sum(axis=0)

data1= data.dropna()
data1.isnull().any()
data1.iloc[6]
data.index[data.isnull().any(axis=1)]
data.iloc[10]
data1.iloc[10]
data.iloc[9]
data.iloc[10]
data.iloc[11]


data2 = data1.select_dtypes(exclude='object')
data2
data2.dtypes
data2.head()

from sklearn.preprocessing import StandardScaler

dt= StandardScaler().fit_transform(data2)
dt

sse=[]
kmeans_kwargs = {'init':'random','n_init':10,'max_iter':300}
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,**kmeans_kwargs).fit(dt)
    sse.append(kmeans.inertia_)
sse
from kneed import KneeLocator
KneeLocator(x=range(1,11),y=sse,curve='convex',direction='decreasing').elbow

plt.plot(range(1,11),sse,marker='*')
plt.grid(True)

kmeans= KMeans(n_clusters=6,n_init=10,max_iter=300).fit(dt)
centroids= kmeans.cluster_centers_
centroids
kmeans.labels_

data2['labels']=kmeans.labels_
data2
data2.sort_values('labels',ascending=True)
data2.labels.value_counts()
kmeans.n_iter_

data2.to_csv('DATA2.csv')

data2
data2['pred']= kmeans.predict(dt)
data2
(data2.labels==data2.pred).sum()

dt


data2
data2.head()
data2.columns
plt.figure(dpi=1000)
scatter =plt.scatter(data2.ApplicantIncome,data2.LoanAmount,c=data2.labels)
handles,labels= scatter.legend_elements(prop='colors')
plt.legend(handles,labels,loc='lower right')


##CT 06

url='https://stackabuse.s3.amazonaws.com/files/hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv'
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
import numpy as np
df = pd.read_csv(url)
#customer_data = pd.read_csv('data/shopping-data.csv')
customer_data = df.copy()
customer_data.head()
#explore data
customer_data
customer_data.head()
customer_data.shape
customer_data.describe()
customer_data.columns

#rename columns
customer_data.columns = ['customerID', 'genre', 'age', 'income', 'spendscore']
customer_data.head()
#Our dataset has five columns: CustomerID, Genre, Age, Annual Income, and Spending Score. To view the results in two-dimensional feature space, we will retain only two of these five columns. We can remove CustomerID column, Genre, and Age column. We will retain the Annual Income (in thousands of dollars) and Spending Score (1-100) columns. The Spending Score column signifies how often a person spends money in a mall on a scale of 1 to 100 with 100 being the highest spender. 

data = customer_data.iloc[:, 2:5]
data


plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))    
plt.show();



from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
pred = cluster.fit_predict(data)
pred

data['labels'] = pred
data
