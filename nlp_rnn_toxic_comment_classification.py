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
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F 

from torchtext.data.utils import get_tokenizer 
from torchtext.vocab import build_vocab_from_iterator

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

for col in target_cols: 
    print(raw_df[col].value_counts(normalize=True))



"""
Prepare the Data for Training 
- Create a vocabulary using TorchText
- Create training and training datasets
- Create PyTorch DataLoaders
"""
tokenizer = get_tokenizer("basic_english") 

#sample_comment = raw_df.comment_text.values[0]
#sample_comment_tokens = tokenizer(sample_comment)

#print(sample_comment_tokens[:20]) 

VOCAB_SIZE = 1500
comment_tokens = raw_df.comment_text.map(tokenizer) 
unc_token = '<unk>'
pad_token = '<pad>'
vocab = build_vocab_from_iterator(iterator=comment_tokens) 
#vocab.set_default_index

print(vocab[["xxxxxxxxxxxx", "rrrrrrrrrrrdd"]])





