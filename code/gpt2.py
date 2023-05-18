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


def main(file):
  df = pd.read_csv(file)
  target = []
  sentence = []
  for item in list(df['sentence']):
    sentence.append(' '.join(item.split()[:-1]))
    target.append(item.split()[-1].replace('.',''))
  df['masked_sentence'] = sentence
  df['target'] = target
  
  loss = []
  with open('output_gpt2.txt','w') as f:
    for i in range(len(df)):
      loss_item = prediction(sentence[i],target[i])
      loss.append(loss_item)
      f.write(f"{sentence[i]} {target[i]} {loss_item} {df['consistent'][i]} {df['kind'][i]} {df['fact'][i]}\n")
  
  df['loss'] = loss
  df.to_csv('output_gpt2.csv')
  return df
main('large_dataset.csv')
