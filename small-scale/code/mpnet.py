import pandas as pd
import torch
import math
#pip install transformers


from transformers import MPNetTokenizer, MPNetForMaskedLM
tokenizer_mpnet = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
model_mpnet = MPNetForMaskedLM.from_pretrained("microsoft/mpnet-base")
model_mpnet.eval()

def prediction(sentence,target_sentence,mask_token='<mask>'):
 
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


def cal_loss(df):
  df['surprisal_original'] = [prediction(df['context_original_mask'][i],df['context_original'][i]) for i in range(len(df))]
  df['surprisal_baseline1'] = [prediction(df['context_baseline1_mask'][i],df['context_baseline1'][i]) for i in range(len(df))]
  df['surprisal_baseline2'] = [prediction(df['context_baseline2_mask'][i],df['context_baseline2'][i]) for i in range(len(df))]
  df['surprisal_baseline3'] = [prediction(df['context_baseline3_mask'][i],df['context_baseline3'][i]) for i in range(len(df))]
  return df

def main(file):
    df = pd.read_csv(file)
    df = cal_surprisal(df)
    return df