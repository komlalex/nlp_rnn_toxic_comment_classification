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



import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F 

#import torchtext
#from torchtext.data.utils import get_tokenizer 
#from torchtext.vocab import build_vocab_from_iterator 

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tqdm.auto as tqdm 

"""Set-up device agnostic code"""
device = "cuda" if torch.cuda.is_available() else "cpu"

"""Download Data"""
raw_df = pd.read_csv("./data/train.csv") 
test_df = pd.read_csv("./data/test.csv")
sub_df = pd.read_csv("./data/sample_submission.csv") 

#print(len(raw_df))
#print(raw_df.info())
#raw_df.obscene.value_counts(normalize=False).plot(kind="bar")
#plt.show() 

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] 

#for col in target_cols: 
#    print(raw_df[col].value_counts(normalize=True))



"""
Prepare the Data for Training 
- Create a vocabulary using sklearn
- Create training and training datasets
- Create PyTorch DataLoaders 
"""




VOCAB_SIZE = 1500
MAX_LENGTH = 150
unk_token = "<unk>"
pad_token = "<pad>"

vectorizer = CountVectorizer(max_features=VOCAB_SIZE,
                            tokenizer= word_tokenize 
                            )


vectorizer.fit(raw_df.comment_text)
vocab = vectorizer.vocabulary_ 
index_to_word = {index: word for word, index in vocab.items()} 
first_word = index_to_word[0]
second_word = index_to_word[1]

ft = vectorizer.get_feature_names_out()

index_to_word[0] = unk_token
index_to_word[1] = pad_token
index_to_word[len(index_to_word)] = first_word
index_to_word[len(index_to_word)] = second_word


word_to_index = {word: index for index, word in index_to_word.items()}


def get_index(word): 
  try:
    return word_to_index[word]
  except:
    return word_to_index[unk_token] 

#raw_df.comment_text.sample(1000).map(tokenizer).map(len).value_counts().plot(kind="hist")
#plt.show()

def pad_tokens(tokens):
  if len(tokens) >= MAX_LENGTH:
    return tokens[:MAX_LENGTH] 
  else: 
    return tokens + [pad_token] * (MAX_LENGTH - len(tokens))


#print(pad_tokens(tokenizer("I love this so so much")))

class JigsawDataset(Dataset):
   def __init__(self, df, is_test = False):
     super().__init__() 
     self.df = df 
     self.is_test = is_test 
   
   def __getitem__(self, index):
    comment_text = self.df.comment_text.values[index].lower()
    comment_tokens = pad_tokens(word_tokenize(comment_text))
    inputs = [get_index(token) for token in comment_tokens]
    inputs = torch.tensor(inputs).float()

    if self.is_test:
      target = torch.tensor([0, 0, 0, 0, 0, 0]).float()
    else: 
      target = torch.from_numpy(self.df[target_cols].values[index]).float()
    return inputs, target

   def __len__(self):
     return len(self.df) 

raw_ds = JigsawDataset(raw_df) 
test_ds = JigsawDataset(test_df, is_test=True)

print(test_ds[15])

""""""






