import time
start_sec = time.time()
import matplotlib.pyplot as plt
from market_snapshot import *
import sys
import numpy as np
import pandas as pd
import os
import datetime
import get_datelist as gd
import get_main as gm

date_list = gd.get()
print date_list
tick_list = gm.get_tick()
tick_size_map = gm.get_tickmap()
volume_map = gm.get_volumemap()

mode_list = ['', '_night']

column_index = ['Pair', 'Date', 'mean', 'variance', 'std/tick_size', 'vol0/vol1', '1% interval', '5% interval', '5% interval(train with 10% data)', '5% interval(train with 20% data)', '0%-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%']
df_content = []

for dl in date_list:
  for ml in mode_list:
    for tl in tick_list:
      target_list = gm.get_main(tl)
      if len(target_list) < 2:
        continue
      file_name ='/root/quant/data/Mid/'+tl+'/'+target_list[0]+target_list[1]+'_'+dl+ml+'_df.pds'
      if os.path.exists(file_name) == False:
        continue
      df = pd.read_csv(file_name)
      diff_list = df['mid_delta'][~np.isnan(df['mid_delta'])]
  
      this_content = []
      this_content.append(target_list[0]+' '+target_list[1])
      date_index = dl+ml
      this_content.append(date_index)
      
      train_rate = 0.9
      train_size = int(len(diff_list)*train_rate)
      
      train_diff_list = diff_list[0:train_size]
      test_diff_list = diff_list[train_size:]
      
      this_mean = round(np.array(diff_list).mean(),2)
      this_var = round(np.array(diff_list).var(), 2)
      this_std = round(np.array(diff_list).std(), 2)
  
      this_mean_10 = round(np.array(diff_list[0:int(len(diff_list)*0.1)]).mean(),2)
      this_var_10 = round(np.array(diff_list[0:int(len(diff_list)*0.1)]).var(), 2)
      this_std_10 = round(np.array(diff_list[0:int(len(diff_list)*0.1)]).std(), 2)
  
      this_mean_20 = round(np.array(diff_list[0:int(len(diff_list)*0.2)]).mean(),2)
      this_var_20 = round(np.array(diff_list[0:int(len(diff_list)*0.2)]).var(), 2)
      this_std_20 = round(np.array(diff_list[0:int(len(diff_list)*0.2)]).std(), 2)
  
      this_content.append(this_mean)
      this_content.append(this_var)
      this_content.append(this_std/tick_size_map[tl])
      volume0 = volume_map[target_list[0]]
      volume1 = volume_map[target_list[1]]
      this_content.append(str(volume0)+':'+str(volume1));
      this_content.append('['+str(this_mean-3*this_std)+', ' + str(this_mean+3*this_std) + ']')
      this_content.append('['+str(this_mean-2*this_std)+', ' + str(this_mean+2*this_std) + ']')
  
      this_content.append('['+str(this_mean_10 - 2*this_std_10)+', ' + str(this_mean_10+2*this_std_10) + ']')
      this_content.append('['+str(this_mean_20 - 2*this_std_20)+', ' + str(this_mean_20+2*this_std_20) + ']')
  
      len_diff = len(diff_list)
      step = int(len_diff/5)
      start = 0
      for i in range(5):
        mean = round(np.array(diff_list[start:start+step]).mean(),2)
        std = round(np.array(diff_list[start:start+step]).std(),2)
        this_content.append('['+str(mean - 2*std)+', ' + str(mean+2*std) + ']')
        start = start + step
      df_content.append(this_content)
      print 'difflist length is ', len(diff_list)
    df = pd.DataFrame(df_content, columns = column_index, index = None)
    df.to_html('/root/quant/data/Mid/mid/all_mid' + dl+ml +'.html')
    
print 'running time is ' + str(time.time()-start_sec)
