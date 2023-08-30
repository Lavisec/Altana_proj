# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:26:35 2023

@author: Lavi Secundo
"""

import numpy as np
import pandas as pd
from scipy.stats import randint

# import matplotlib.pyplot as plt
# from pycaret.classification import *

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

#%% load the data
fn = r'C:\Documents\Personal\Altana\DataSets\ds-project-train.csv'
train_df = pd.read_csv(fn)
fn = r'C:\Documents\Personal\Altana\DataSets\ds-project-validation.csv'
validate_df = pd.read_csv(fn)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
#%% label the train and validate and combine for cleaning 
train_df['TrainTest'] = 'Train'
validate_df['TrainTest'] = 'Validate'
all_df = pd.concat([train_df, validate_df])
#%% clean the data
all_df['SHIPPER'][all_df['SHIPPER'].isna()] = 'UnKnown' #replace empty SHIPPER with UnKnown
all_df['NOTIFY'][all_df['NOTIFY'].isna()] = 'UnKnown' #replace empty NOTIFY with UnKnown
all_df['US.PORT'][all_df['US.PORT'].isna()] = 'UnKnown' #replace empty PORT with UnKnown
all_df['US.PORT'] = all_df['US.PORT'].str.lower().str.split(',').str.get(0) #keep only port city in lower cases
#%% clean the data by removing rows 
all_df = all_df[~all_df['COUNTRY.OF.ORIGIN'].isna()] #remove rows with empty COUNTRY.OF.ORIGIN
all_df = all_df[~all_df['PRODUCT.DETAILS'].isna()] #remove rows with empty PRODUCT.DETAILS
all_df = all_df[~all_df['ARRIVAL.DATE'].isna()] #remove rows with empty ARRIVAL.DATE
all_df = all_df[all_df['US.PORT'] !='unknown'] #remove rows with unknown US.PORT

all_df['ARRIVAL.DATE'] = pd.to_datetime(all_df['ARRIVAL.DATE'])
all_df['dayOfYear'] = all_df['ARRIVAL.DATE'].dt.dayofyear -1
all_df['Sin_dayofyear'] = np.sin(all_df['dayOfYear']/365*2*np.pi) #calculate sin of day of the year
#%% reduce the size of teh data to make things work faster
N = 10
all_df = all_df.iloc[::N, :]
all_df.reset_index(inplace=True)
#%% claen the 'PRODUCT.DETAILS' 
all_df['PRODUCT.DETAILS'] = all_df['PRODUCT.DETAILS'].str.lower()
all_df['PRODUCT.DETAILS'] = all_df['PRODUCT.DETAILS'].str.replace('\W',' ') # replace all non alphanumeric with spaces
all_df['PRODUCT.DETAILS'] = all_df['PRODUCT.DETAILS'].str.replace(r'(\d)([A-Za-z])',r'\1 \2', regex=True) # add space between number and text
all_df['PRODUCT.DETAILS'] = all_df['PRODUCT.DETAILS'].str.replace(r'([A-Za-z])(\d)',r'\1 \2', regex=True) # add space between text and number
all_df['PRODUCT.DETAILS'] = all_df['PRODUCT.DETAILS'].str.replace(r' +',r' ', regex=True) # replace multi space with single space
#%% claen the 'PRODUCT.DETAILS' by removing short, frequent, and infrequent words
auxWords = all_df['PRODUCT.DETAILS'].str.split(' ').explode().tolist() # create a list of all the words in 'PRODUCT.DETAILS
wordList_toRem = pd.Series(auxWords).value_counts().to_frame(name="wordFreq") # create a df with unique words list and their frequency
wordList_toRem = wordList_toRem[wordList_toRem.index!=''] # remove empty words
wordList_toRem['wordLen'] = wordList_toRem.index.str.len() # calculate word length

wordList_toRem['lenLess2'] = wordList_toRem['wordLen'] <2
wordList_toRem['freqLess10'] = wordList_toRem['wordFreq'] < 10
wordList_toRem['freqMore10000'] = wordList_toRem['wordFreq'] > 10000
wordList_toRem = wordList_toRem[wordList_toRem[['lenLess2','freqLess10','freqMore10000' ]].any(axis = 1)] # remove short, frequent, and infrequent words
wordList_toRem=wordList_toRem.index # generate a list of words to remove form 'PRODUCT.DETAILS'
#%% remove a list of words  form 'PRODUCT.DETAILS'
pat = r'\b(?:{})\b'.format('|'.join(wordList_toRem))
all_df['PRODUCT.DETAILS_clean'] = all_df['PRODUCT.DETAILS'].str.replace(pat, ' ')
all_df['PRODUCT.DETAILS_clean'] = all_df['PRODUCT.DETAILS_clean'].str.replace(r' +',r' ', regex=True).str.strip() # replace multi space with single space and remove whitespace 
#%% categorize PRODUCT.DETAILS to 100 classes 
corpus=all_df['PRODUCT.DETAILS_clean'].values.tolist()
corpus_embeddings = embedder.encode(corpus)
num_clusters = 100
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
all_df['PRODUCT_CLASS_NUM']=clustering_model.labels_
#%% replace categorical strings with numbers 
all_df1 = all_df.copy()
all_df1['COUNTRY_num_N'] = all_df1['COUNTRY.OF.ORIGIN'].astype('category').cat.codes
all_df1['PORT_num_N'] = all_df1['US.PORT'].astype('category').cat.codes
all_df1['VESSEL_num_N'] = all_df1['VESSEL.NAME'].astype('category').cat.codes

# keep only some features
all_df1 = all_df1[['Sin_dayofyear','WEIGHT..KG.','PORT_num_N','VESSEL_num_N','PRODUCT_CLASS_NUM','COUNTRY_num_N','TrainTest']]

#%% split datat back to train and test
X_train = all_df1[all_df1['TrainTest'] == 'Train']
y_train = all_df1[all_df1['TrainTest'] == 'Train']['COUNTRY_num_N']
X_train = X_train.drop(['COUNTRY_num_N', 'TrainTest'], axis=1)

X_test = all_df1[all_df1['TrainTest'] == 'Validate']
y_test = all_df1[all_df1['TrainTest'] == 'Validate']['COUNTRY_num_N']
X_test = X_test.drop(['COUNTRY_num_N', 'TrainTest'], axis=1)

#%% run the RF classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%% tune hyperparameters
param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,  param_distributions = param_dist,  n_iter=5, cv=5)
# Fit the random search object to the data
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred,average = 'weighted')
print("precision:", precision)

recall = recall_score(y_test, y_pred, average = 'weighted')
print("recall:", recall)

f1 = f1_score(y_test, y_pred, average = 'weighted')
print("f1:", f1)
# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

