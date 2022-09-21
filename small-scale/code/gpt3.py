import sys
import csv
import json
import requests
import pandas as pd
import torch
import math

###estimate surprisal via GPT-3
def prediction(text, target, write_text=False, write_metadata=True):
    headers = {"Authorization": "Bearer [specify token number here]"}
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
        if (len(text) - len(target) -1) == result['offset']:
          flag = 1
        if flag == 1:
          surprisal = surprisal - result['logprob']
        #print(result)
    return surprisal

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
