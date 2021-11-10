import pandas as pd
import os
import pandas as pd
import numpy as np
import math
from ctgan import CTGANSynthesizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import smote_variants as sv
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from table_evaluator import load_data, TableEvaluator
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

#import dataset
data=pd.read_csv(r'PathUrl\hmeq.csv')

#pre-deal
for i in range(len(data['MORTDUE'])):
    if pd.isnull(data['MORTDUE'].values[i]):
        data['MORTDUE'].values[i]=73760
    if pd.isnull(data['VALUE'].values[i]):
        data['VALUE'].values[i]=101776
    if pd.isnull(data['REASON'].values[i]):
        data['REASON'].values[i]='HomeImp'
    if pd.isnull(data['JOB'].values[i]):
        data['JOB'].values[i]='other'
    if pd.isnull(data['YOJ'].values[i]):
        data['YOJ'].values[i]=9
    if pd.isnull(data['DEROG'].values[i]):
        data['DEROG'].values[i]=0
    if pd.isnull(data['DELINQ'].values[i]):
        data['DELINQ'].values[i]=0
    if pd.isnull(data['CLAGE'].values[i]):
        data['CLAGE'].values[i]=180
    if pd.isnull(data['NINQ'].values[i]):
        data['NINQ'].values[i]=0
    if pd.isnull(data['CLNO'].values[i]):
        data['CLNO'].values[i]=21
    if pd.isnull(data['DEBTINC'].values[i]):
        data['DEBTINC'].values[i]=33.7799

#Split the data set into training sets and tests
target=data['BAD'].values.astype('float')
data_list=data.drop(['BAD'],axis=1).values
global X_train_data, X_test_data,y_train_data, y_test_data
kf = KFold(n_splits=5,shuffle=True,random_state=10)
for train_index , test_index in kf.split(data_list):
    X_train_data, X_test_data = data_list[train_index], data_list[test_index]
    y_train_data, y_test_data = target[train_index], target[test_index]

#Collating training set
data_columns=data.columns
data_columns=list(data_columns)
del data_columns[0]
train_data=pd.DataFrame(X_train_data)
train_data.columns=data_columns
train_data['BAD']=y_train_data
train_data[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]=train_data[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']].apply(pd.to_numeric)

#encode
le = LabelEncoder()
le_count = 0
# Iterate through the columns
for col in train_data:
    if train_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(train_data[col])
            # Transform both training and testing data
            train_data[col] = le.transform(train_data[col])
            # Keep track of how many columns were label encoded
            le_count += 1
encode_data = pd.get_dummies(train_data)
encode_data_columns=encode_data.columns
encode_data_columns=list(encode_data.columns)

#clustering
kmeans_train_x=encode_data[encode_data_columns]
kmeans = KMeans(n_clusters=5)
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(kmeans_train_x)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((train_data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'clusters'},axis=1,inplace=True)

#Get header
data_columns=cluster_data1.columns
data_columns=list(data_columns)
del data_columns[0]
del data_columns[-1]

#Assign different quantities to each category according to the clustering results,
sample_list=[543,551,267,694,442]

final_data=train_data

#Initially set up five classes
for  i in range(5):
  cluster_data1=result[result['clusters']==i]
  cluster_data1=cluster_data1[cluster_data1['BAD']>0]
  data_list=cluster_data1.drop(['BAD','clusters'],axis=1)
  ctgan = CTGANSynthesizer()
  ctgan.fit(data_list, data_columns, epochs=30)
  samples0 = ctgan.sample(sample_list[i])
  samples0['BAD']=1
  final_data=pd.concat([final_data,sanples0],axis=0)

le = LabelEncoder()
le_count = 0

for col in final_data:
    if final_data[col].dtype == 'object':
        if len(list(final_data[col].unique())) <= 2:
            le.fit(final_data[col])
            final_data[col] = le.transform(final_data[col])
            le_count += 1
final_data = pd.get_dummies(final_data)

#Collating testing set
test_data=pd.DataFrame(X_test_data)
test_data.columns=data_columns
test_data['BAD']=y_test_data
test_data[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]=test_data[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']].apply(pd.to_numeric)

le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in test_data:
    if test_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(test_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(test_data[col])
            # Transform both training and testing data
            test_data[col] = le.transform(test_data[col])
            # Keep track of how many columns were label encoded
            le_count += 1
test_data = pd.get_dummies(test_data)

y_train=final_data['BAD'].values.astype('float')
X_train=final_data.drop(['BAD'],axis=1).values.astype('float')
y_test=test_data['BAD'].values.astype('float')
X_test=test_data.drop(['BAD'],axis=1).values.astype('float')

# format Dataset 
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'l2', 'auc'},  
    'num_leaves': 31,  
    'learning_rate': 0.05,  
    'feature_fraction': 0.9,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 5,  
    'verbose': 1  
}
gbm = lgb.train(params, lgb_train, num_boost_round=500,valid_sets=None)

y_test_pred=gbm.predict(X_test)
for i in range(len(y_test_pred)):
    if y_test_pred[i]>0.5:
        y_test_pred[i]=1
    else:
        y_test_pred[i]=0
print('f1_score:',metrics.f1_score(y_test,y_test_pred) )
recall2=metrics.recall_score(y_test,y_test_pred)
tn2, fp2, fn2, tp2 = metrics.confusion_matrix(y_test, y_test_pred).ravel()
specificity2 = tn2 / (tn2+fp2)
print('g_mean:',np.sqrt(recall2*specificity2))




