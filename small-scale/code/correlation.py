import pandas as pd
from scipy.stats import pearsonr 

def correlation(file):
  df = pd.read_csv(file)

  rw_con = df[df['condition']=='rw_con'].reset_index()
  rw_incon = df[df['condition']=='rw_incon'].reset_index()
  cw_con = df[df['condition']=='cw_con'].reset_index()
  cw_incon = df[df['condition']=='cw_incon'].reset_index()
  answer_rw = [rw_con['surprisal_original'][i] < rw_incon['surprisal_original'][i] for i in range(len(rw_con))]
  knowledge_rw = [rw_con['surprisal_baseline1'][i] - rw_incon['surprisal_baseline1'][i] for i in range(len(rw_con))]

  answer_cw = [cw_con['surprisal_original'][i] < cw_incon['surprisal_original'][i] for i in range(len(cw_con))]
  knowledge_cw = [-cw_con['surprisal_baseline1'][i] + cw_incon['surprisal_baseline1'][i] for i in range(len(cw_con))]
  print('rw',pearsonr(answer_rw,knowledge_rw))
  print('cw',pearsonr(answer_cw,knowledge_cw))

  correlation('common_mpnet.csv')