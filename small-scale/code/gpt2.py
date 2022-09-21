import pandas as pd
import torch
import math
#pip install transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#get GPT2 probability
def prediction(text,target):
    #reformat text
    text = text.split(target)[0]
    target = ' '+target
    tokenized_text = tokenizer.tokenize(text)
    tokenized_target = tokenizer.tokenize(target)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    target_index =  tokenizer.convert_tokens_to_ids(tokenized_target)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        predictions = model(tokens_tensor)
    prob_all = torch.softmax(predictions[0][0,-1,:],dim=0)

    prob = 0
    for idx in target_index:
      prob = prob + prob_all[idx].item()
    prob = prob/len(target_index)
    #predicted_score = predictions[0][0,0,target_index].item()
    return -math.log(prob)


def cal_surprisal(df):
  df['surprisal_original'] = [prediction(df['context_original'][i],df['target'][i]) for i in range(len(df))]
  df['surprisal_baseline1'] = [prediction(df['context_baseline1'][i],df['target'][i]) for i in range(len(df))]
  df['surprisal_baseline2'] = [prediction(df['context_baseline2'][i],df['target'][i]) for i in range(len(df))]
  df['surprisal_baseline3'] = [prediction(df['context_baseline3'][i],df['target'][i]) for i in range(len(df))]
  #df['surprisal_baseline4'] = [prediction(df['context_baseline4'][i],df['target'][i]) for i in range(len(df))]
  return df

def main(file):
    df = pd.read_csv(file)
    df = cal_surprisal(df)
    return df
