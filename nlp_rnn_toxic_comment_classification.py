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
- Create a vocabulary using TorchText
- Create training and training datasets
- Create PyTorch DataLoaders
"""


#sample_comment = raw_df.comment_text.values[0]
#sample_comment_tokens = tokenizer(sample_comment)

#print(sample_comment_tokens[:20]) 

VOCAB_SIZE = 200
"""
comment_tokens = raw_df.comment_text.map(tokenizer) 
unc_token = '<unk>'
pad_token = '<pad>'

#print(vocab["this"])

sample_comment = raw_df.comment_text.values[0]
sample_comment_tokens = tokenizer(sample_comment)
sample_comment_tokens = sample_comment_tokens[:10]
#print(sample_comment_tokens)


vocab.set_default_index(vocab[unc_token])
sample_indices = vocab.lookup_indices(sample_comment_tokens) 

sample_comment_recovered = vocab.lookup_tokens(sample_indices)
#print(sample_comment_recovered) 
"""

"""USING SKLEARN"""
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
print(type(vocab), type(ft))
print(ft[:10])

index_to_word[0] = unk_token
index_to_word[1] = pad_token
index_to_word[len(index_to_word)] = first_word
index_to_word[len(index_to_word)] = second_word


word_to_index = {word: index for index, word in index_to_word.items()}

abi_word = "fhberthgjrtnsj"

def get_index(word): 
  try:
    return word_to_index[word]
  except:
    return word_to_index[unk_token] 

print("nonsense word", get_index(abi_word))
"""Create Training & Validation Sets 
- Define a custom PyTorch Dataset
- Pass raw data into the dataset
- Split the PyTorch Dataset
"""
#raw_df.comment_text.sample(1000).map(tokenizer).map(len).value_counts().plot(kind="hist")
#plt.show()
MAX_LENGTH = 150

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

print(raw_ds[12]) 

print(raw_df[target_cols].values[12])






