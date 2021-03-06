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
import self_input as si

date,mode=si.init()
target_list=['ni1905', 'ni1903']

column_index = ['Date', 'mean', 'variance', '1% interval', '5% interval', '5% interval(train with 10% data)', '5% interval(train with 20% data)', '0%-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%']
df_content = []

file_name ='/root/quant/data/Mid/'+target_list[0]+target_list[1]+'_'+date+mode+'_df.pds'
if os.path.exists(file_name) == False:
  print file_name + ' not exist'
  sys.exit(1)
df = pd.read_csv(file_name)
diff_list = df['mid_delta'][~np.isnan(df['mid_delta'])]

this_content = []
date_index = date+mode
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
df.to_html('/root/quant/data/Mid/mid_analyse' + target_list[0] + target_list[1] + date+mode +'.html')
    
print 'running time is ' + str(time.time()-start_sec)
