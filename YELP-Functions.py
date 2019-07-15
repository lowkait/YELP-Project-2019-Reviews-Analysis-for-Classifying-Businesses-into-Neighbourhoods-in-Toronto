#!/usr/bin/env python
# coding: utf-8

# # YELP Project 2019: Reviews Analysis for Classifying Businesses into Neighbourhoods in Toronto
# ### Overall Question(s): Can language distinguish groups of businesses/people? What in the language? Why?

# In[3]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from tqdm import tqdm
from collections import *
import operator
import itertools
import dill
import geopandas as gpd
import matplotlib.pyplot as plt
import descartes
from shapely.geometry import Point, Polygon
import math
from scipy import sparse
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 500)

def everyWord(file, stop_words):
    everyWord_counter = Counter()
    for comment in tqdm(file['1'].values):
        comment = [t.lower() for t in comment.replace('.',' ').replace(',',' ').split(" ") if ((len(t) > 1) and (t.lower() not in stop_words))]
        everyWord_counter.update(comment)
    return everyWord_counter

def topktermFreq(everyWord, stop_words, k):
    AllWords_stop = Counter(everyWord)
    sorted_dict = OrderedDict(sorted(AllWords_stop.items(), key=operator.itemgetter(1), reverse=True))
    topktermFreq = dict(itertools.islice(sorted_dict.items(), k))
    return topktermFreq

def topkdocFreq(file, stop_words, k):
    allWordsinFile = everyWord(file, stop_words)
    topkTF = topktermFreq(allWordsinFile, stop_words, k)
    topkdocFreq = defaultdict(int)
    for comment in tqdm(file['1'].values):
        comment = set(t.lower() for t in comment.replace('.',' ').replace(',',' ').split(" ") if len(t) > 1 and (t.lower() not in stop_words))
        for word in comment:
            if word in topkTF.keys():
                topkdocFreq[word] += 1
            else:
                continue
    return topkdocFreq

def TFIDF_xtrain(xtrain_file, topkDF, stop_words):
    row = []
    col = []
    data = []
    # for each of the comments (rows)
    row_index = 0
    # loop through each of the comments in the 
    for comment in tqdm(xtrain_file['1'].values):
        #comment contains all the words in the comments, but we are only interested in the 15
        comment = [t.lower() for t in comment.replace('.',' ').replace(',',' ').split(" ") if ((len(t) > 1) and (t.lower() not in stop_words))]
        # create a dictionary for all words
        c_counter = Counter(comment)        
        col_index = 0
        # loop through the top 10k words
        for word in topkDF.keys():
            if word in c_counter.keys():
                row.append(row_index)
                col.append(col_index)
                data.append(round(((c_counter[word]/len(comment))*math.log10(len(xtrain_file)/topkDF[word])),5))
                col_index += 1
            else:
                col_index += 1
        row_index += 1
    return sparse.coo_matrix((data,(row,col)), shape = (len(xtrain_file),len(topkDF))).toarray()

def TFIDF_test(xtrain_file, test_file, topkDF, stop_words):
    row = []
    col = []
    data = []
    # for each of the comments (rows)
    row_index = 0
    # loop through each of the comments in the 
    for comment in tqdm(test_file['1'].values):      
        #comment contains all the words in the comments, but we are only interested in the 15
        comment = [t.lower() for t in comment.replace('.',' ').replace(',',' ').split(" ") if ((len(t) > 1) and (t.lower() not in stop_words))]
        # create a dictionary for all words
        c_counter = Counter(comment)  
        col_index = 0
        # loop through the top 10k words
        for word in topkDF.keys():
            if word in c_counter.keys():
                row.append(row_index)
                col.append(col_index)
                data.append(round(((c_counter[word]/len(comment))*math.log10(len(xtrain_file)/topkDF[word])),5))
                col_index += 1
            else:
                col_index += 1
        row_index += 1
    return sparse.coo_matrix((data,(row,col)), shape = (len(test_file),len(topkDF))).toarray()


def existing_neighbourhood_dictionary(businesses_file, neighbourhoods_file):
    id=[]
    latitude=[]
    longitude=[]
    for x in range(businesses_file.shape[0]):
        id.append(businesses_file.iloc[x,13])
        longitude.append(businesses_file.iloc[x,30])
        latitude.append(businesses_file.iloc[x,31])  
    df = pd.DataFrame(
        {'ID': id,
         'Latitude': latitude,
         'Longitude': longitude})
    gdf = gpd.GeoDataFrame(df, geometry= gpd.points_from_xy(df.Longitude, df.Latitude))    
    business_neighbourhood = {}
    b_names = businesses_file.iloc[:,13]
    for key in b_names:
        business_neighbourhood[key] = float('nan')      
    for i in range(gdf.shape[0]): #for businesses
        for j in range(neighbourhoods_file.shape[0]): # for neighbourhood
            if (neighbourhoods_file.loc[j, 'geometry']).contains(gdf.iloc[i,3]) == True:
                business_neighbourhood[businesses_file.iloc[i,13]] = neighbourhoods_file.iloc[j,6]
    return business_neighbourhood

def official_neighbourhoods(businesses_file):
    neighbourhood_official = {}
    for x in tqdm(range(businesses_file.shape[0])):
        neighbourhood_official[(businesses_file.iloc[x,13])] = businesses_file.iloc[x,29]
    return neighbourhood_official

def cosineSimilarity(tfidf_xtrain, tfidf_test):
    similarities = np.zeros((len(tfidf_test),len(tfidf_xtrain)))
    for i in tqdm(range(tfidf_test.shape[0])):
        a = tfidf_test[i]
        for j in range(tfidf_xtrain.shape[0]):
                b = tfidf_xtrain[j]
                cosinesimilarity = round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 1)
                if cosinesimilarity == 0:
                    continue
                else:
                    similarities[i][j] = cosinesimilarity
    return similarities

def kNearestNeighbours(test_file, xtrain_comments_file, existing_neighbourhoods_xtrain, cos_sim, k): #want to specify the size  
    kNearest = {}   
    # Have a dictionary stating the business and the neighbourhood, for ex. {2: 'A', 3: 'C', 6: 'C'}
    for row in tqdm(range(len(cos_sim))):
        possible_neighbours = []
        for col in range(len(cos_sim[0])):
            possible_neighbours.append((cos_sim[row][col], existing_neighbourhoods_xtrain[xtrain_comments_file.iloc[col,1]]))
        possible_neighbours.sort(key=lambda x: x[0], reverse=True)
  
        votes = Counter()
        for index in range(k):
            votes[possible_neighbours[index][1]] += 1
    
        kNearest[test_file.iloc[row,1]] = votes.most_common(1)[0][0]
    return kNearest
        
def Accuracy(kNearest_, existing_neighbourhood_dictionary):
    total = len(kNearest_)
    sum = 0
    for key in kNearest_.keys():
        if kNearest_[key] == existing_neighbourhood_dictionary[key]:
            sum += 1
    average = round(sum/total,5)
    return average

