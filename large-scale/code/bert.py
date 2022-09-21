import pandas as pd
import torch
import math
#pip install transformers
from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# from transformers import RobertaTokenizer, RobertaForMaskedLM
# tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
# model_roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
# model_roberta.eval()

# from transformers import MPNetTokenizer, MPNetForMaskedLM
# tokenizer_mpnet = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
# model_mpnet = MPNetForMaskedLM.from_pretrained("microsoft/mpnet-base")
# model_mpnet.eval()

def prediction(sentence,target_sentence,mask_token='[MASK]'):
  labels = tokenizer(target_sentence, return_tensors="pt")["input_ids"]
  #target_length = len(tokenizer(sentence, return_tensors="pt")["input_ids"][0])
  inputs = tokenizer(sentence, return_tensors="pt")
  target_length = len(labels[0])
  input_length = len(inputs['input_ids'][0])
  if target_length != input_length:
    mask = mask_token
    for i in range(target_length-input_length):
      mask = mask + mask_token
    sentence = sentence.replace(mask_token,mask)
    #print(sentence)
    inputs = tokenizer(sentence, return_tensors="pt")
  outputs = model(**inputs, labels=labels)
  loss = outputs.loss
  return loss.item()

def main(file):
  df = pd.read_csv(file)
  target_sentence = list(df['sentence'])
  sentence = []
  for item in target_sentence:
    sentence.append(item.replace(item.split()[-1],'[MASK].'))
  df['masked_sentence'] = sentence
  loss = []
  with open('output.txt','w') as f:
    for i in range(len(df)):
      loss_item = prediction(sentence[i],target_sentence[i])
      loss.append(loss_item)
      f.write(f"{sentence[i]} {target_sentence[i]} {loss_item} {df['consistent'][i]} {df['kind'][i]} {df['fact'][i]}\n")
  
  df['loss'] = loss
  df.to_csv('output.csv')
  return df
main('counterfactual_large_dataset.csv')
