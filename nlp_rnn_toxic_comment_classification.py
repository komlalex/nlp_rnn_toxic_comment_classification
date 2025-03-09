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
import torch.nn as nn 
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader 
from torch.utils.data import random_split 

import pytorch_lightning as pl


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
    inputs = torch.tensor(inputs).long()

    if self.is_test:
      target = torch.tensor([0, 0, 0, 0, 0, 0]).float()
    else: 
      target = torch.from_numpy(self.df[target_cols].values[index]).float()
    return inputs, target

   def __len__(self):
     return len(self.df) 

raw_ds = JigsawDataset(raw_df) 
test_ds = JigsawDataset(test_df, is_test=True)

print(raw_ds[0])

"""Create Test and Validation Sets"""
VAL_FRAC = 0.25 
train_ds, val_ds = random_split(raw_ds, [1-VAL_FRAC, VAL_FRAC])

"""Create PyTorch DataLoaders"""
BATCH_SIZE = 256
train_dl = DataLoader(train_ds, 
                      shuffle=True, 
                      batch_size=BATCH_SIZE,
                      pin_memory=True, 
                      )
val_ds = DataLoader(
                    val_ds, 
                    batch_size=BATCH_SIZE * 2, 
                    pin_memory=True,
                   
                    )
test_dl = DataLoader(test_ds, 
                      batch_size=BATCH_SIZE * 2, 
                      pin_memory=True, 
                
                    )  

for batch in train_dl: 
  b_inputs, b_targets = batch 
  print("b_input.shape", b_inputs.shape)
  print("b_targets.shape", b_targets.shape)
  break 

"""
Build a Recurrent Neural Network
1. Understand how recurrent neural networks work
2. Create a recurrent neural network 
3. Pass some data through the network
"""
device = "cuda" if torch.cuda.is_available() else "cpu" 


emb_layer = nn.Embedding(len(word_to_index), embedding_dim=256, padding_idx=1)
rnn_layer = nn.RNN(256, 128, 1, batch_first=True)

for batch in train_dl: 
  x, y = batch 
  print(f"x shape {x.shape}")
  print(f"y shape", y.shape)

  emb_out = emb_layer(x)
  print(f"emb_out shape: {emb_out.shape}") 

  rnn_out, hn = rnn_layer(emb_out)
  print(f"rnn_out shape: {rnn_out.shape}")
  print(f"hn shape: {hn.shape}")
  break 

class JigsawModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(len(word_to_index), 256, 1)
    self.lstm = nn.LSTM(256, 128, 1, batch_first=True)
    self.linear = nn.Linear(128, 6)
    self.learning_rate = 0.001

  def forward(self, x: torch.Tensor) -> torch.Tensor: 
    out = self.emb(x)
    out, hn = self.lstm(out)
    out = F.relu(out[:, -1, :]) 
    out = self.linear(out) 
    return out 
  
  def training_step(self, batch, batch_idx):
    x, y = batch 
    x, y = x.to(device), y.to(device)
    outputs = self(x) 
    loss = F.binary_cross_entropy_with_logits(outputs, y) 
    return loss 

  def validation_step(self, batch, batch_idx):
    x, y = batch 
    x, y = x.to(device), y.to(device)
    outputs = self(x)
    loss = F.binary_cross_entropy_with_logits(outputs, y) 
    return loss.item()

  def on_validation_epoch_end(self, validation_step_outputs):
    loss = torch.mean(validation_step_outputs)
    print(f"Epoch: {self.current_epoch} | Loss: {loss}") 

  def predict_step(self, batch, batch_idx):
    x, y = batch 
    x, y = x.to(device), y.to(device) 
    outputs = self(x)
    probs  = torch.sigmoid(outputs) 
    return probs 


  def configure_optimizers(self):
    return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
    
    

model = JigsawModel().to(device) 

for batch in train_dl: 
  x, y = batch 
  x, y = x.to(device), y.to(device) 
  print(f"x shape {x.shape}")
  print(f"y shape", y.shape)

  outputs = model(x) 
  print(f"Output shape: {outputs.shape}") 

  probs = torch.sigmoid(outputs)
  loss = F.binary_cross_entropy_with_logits(probs, y)
  print(f"Loss: {loss}")
  break 

"""Train and Evaluate the Model"""
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_dl)


