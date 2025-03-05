"""
NATURAL LANGUAGE PROCESSING WITH RNNs - Toxic Comment Classification

OUTLINE:
- Download and explore data
- Prepare the data for training 
- Build a recurrent neural network
- Train & evaluate our model
""" 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, f1_score 

import torch
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import tqdm.auto as tqdm 


"""Download Data"""
raw_df = pd.read_csv("./data/train.csv") 
test_df = pd.read_csv("./data/test.csv")
sub_df = pd.read_csv("./data/sample_submission.csv") 

print(len(raw_df))
#raw_df.obscene.value_counts(normalize=False).plot(kind="bar")
#plt.show()

stemmer = SnowballStemmer(language="english")
english_stopwords = stopwords.words("english")

def tokenize(text): 
    return [stemmer.stem(token) for token in word_tokenize(text)] 

vectorizer = TfidfVectorizer(
    lowercase=True,
    tokenizer=tokenize, 
    max_features=2000,
    stop_words=english_stopwords
) 

vectorizer.fit(raw_df.comment_text) 

print(vectorizer.get_feature_names_out())