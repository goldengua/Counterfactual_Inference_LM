
import sys
import csv
import json
import requests
import pandas as pd
import math

def prediction(text, target, write_text=False, write_metadata=True):
    headers = {"Authorization": "Bearer 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"}
    r = requests.get("https://api.openai.com/v1/engines/davinci/completions/browser_stream",
        headers=headers,
        params={
            'prompt': text,
            'max_tokens': 0,
            'logprobs': 0,
            'echo': True,
        }
    )
    line, *_ = r.text.splitlines()
    try:
        data = json.loads(line.strip("data: "))
    except json.decoder.JSONDecodeError:
        print(r.text, file=sys.stderr)
        return
    logprobs = data['choices'][0]['logprobs']['token_logprobs']
    tokens = data['choices'][0]['logprobs']['tokens']
    offsets = data['choices'][0]['logprobs']['text_offset']
    model = data['model']
    text = data['choices'][0]['text']
    time = data['created']
    id = data['id']
    surprisal = 0
    flag = 0
    for token, logprob, offset in zip(tokens, logprobs, offsets):
        result = {
            'token': token,
            'logprob': logprob,
            'offset':offset
        }
        if ((len(text) - len(target) -1) == result['offset']):
          flag = 1
        if flag == 1 and result['token']!='.':
          surprisal = surprisal - result['logprob']
        #print(result)
    return surprisal

def main(file):
  df = pd.read_csv(file)
  sentence = df['sentence']
  target = []
  
  for item in sentence:
    target.append(' '+item.split()[-1].replace('.',''))
    
  df['target'] = target
  loss = []
  with open('output_gpt3.txt','w') as f:
    for i in range(len(df)):
      loss_item = prediction(sentence[i],target[i])
      loss.append(loss_item)
      print(f"{sentence[i]} {target[i]} {loss_item} {df['consistent'][i]} {df['kind'][i]} {df['fact'][i]}\n")
      f.write(f"{sentence[i]} {target[i]} {loss_item} {df['consistent'][i]} {df['kind'][i]} {df['fact'][i]}\n")
  
  df['loss'] = loss
  df.to_csv('output_gpt3.csv')
  return df
main('large_dataset.csv')

