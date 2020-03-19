from common.order import *
from common.market_snapshot import *
import struct
import os
import pandas as pd
from algo import *
import sys

class Reader:
  def __init__(self):
    self.shot_fmt =  "32s5d5d5i5id2i2d1?2Q?7s"
    self.strat_fmt = "32s5d5d5i5id2i2d1?2Q?7s"
    self.order_fmt = "4Q32sd3i32s3i128s"
    self.ex_fmt = "2Qi32sid64si"
    self.shot_size = struct.calcsize(self.shot_fmt)
    self.order_size = struct.calcsize(self.order_fmt)
    self.strat_size = struct.calcsize(self.strat_fmt)
    self.ex_size = struct.calcsize(self.ex_fmt)

  def get_ordersize(self):
    return self.order_struct_size

  def get_shotsize(self):
    return self.shot_struct_size

  def get_stratsize(self):
    return self.strat_struct_size

  def load_order_file(self, file_path):
    self.order_content = open(file_path).read()
    if len(self.order_content) % self.order_size != 0:
      print("%s filesize not order's times" %(file_path))
      #sys.exit(1)
    self.order_struct_size = int(len(self.order_content) / self.order_size)

  def load_shot_file(self, file_path):
    self.shot_content = open(file_path).read()
    if len(self.shot_content) % self.shot_size != 0:
      print("%s filesize not shot's times" %(file_path))
      #sys.exit(1)
    self.shot_struct_size = int(len(self.shot_content) / self.shot_size)

  def load_ex_file(self, file_path):
    self.ex_content = open(file_path).read()
    if len(self.ex_content) % self.ex_size != 0:
      print("%s filesize not shot's times" %(file_path))
      #sys.exit(1)
    self.ex_struct_size = int(len(self.ex_content) / self.ex_size)

  def load_strat_file(self, file_path):
    self.strat_content = open(file_path).read()
    if len(self.strat_content) % self.strat_size != 0:
      print("%s filesize not strat shot's times" %(file_path))
      #sys.exit(1)
    self.strat_struct_size = int(len(self.strat_content) / self.strat_size)

  def read_bshot(self, i):
    if i >= self.shot_struct_size:
      print("oversize for shot %i > %i" %(i, self.shot_struct_size))
      return MarketSnapshot()
    content = self.shot_content[i*self.shot_size:(i+1)*self.shot_size]
    #print(len(struct.unpack(self.shot_fmt, content)))
    shot = MarketSnapshot()
    _, time_sec, time_usec='',0,0
    shot.ticker, shot.bids[0], shot.bids[1], shot.bids[2], shot.bids[3], shot.bids[4], shot.asks[0], shot.asks[1], shot.asks[2], shot.asks[3], shot.asks[4], shot.bid_sizes[0], shot.bid_sizes[1], shot.bid_sizes[2], shot.bid_sizes[3], shot.bid_sizes[4], shot.ask_sizes[0], shot.ask_sizes[1], shot.ask_sizes[2], shot.ask_sizes[3], shot.ask_sizes[4], shot.last_trade, shot.last_trade_size, shot.volume, shot.turnover, shot.open_interest, _, time_sec, time_usec, shot.is_initialized, res = struct.unpack(self.shot_fmt, content)
    shot.time = int(time_sec) + float("0."+str(time_usec))
    return shot.Filter()

  def read_bex(self, i):
    if i >= self.ex_struct_size:
      print("oversize for ex %i > %i" %(i, self.ex_struct_size))
      return ExchangeInfo()
    content = self.ex_content[i*self.ex_size:(i+1)*self.ex_size]
    #print(len(struct.unpack(self.shot_fmt, content)))
    ex = ExchangeInfo()
    shot_sec, shot_usec, ex.type, ex.ticker, ex.order_ref, ex.trade_size, ex.trade_price, ex.reason, ex.side = struct.unpack(self.ex_fmt, content)
    ex.time_str = int(shot_sec) + float("0."+str(shot_usec))
    return ex

  def read_bstrat(self, i):
    if i >= self.strat_struct_size:
      print("oversize for strat %i > %i" %(i, self.strat_struct_size))
      return MarketSnapshot()
    content = self.strat_content[i*self.strat_size:(i+1)*self.strat_size]
    # print(len(struct.unpack(self.strat_fmt, content)))
    strat = MarketSnapshot()
    _, time_sec, time_usec='',0,0
    strat.ticker, strat.bids[0], strat.bids[1], strat.bids[2], strat.bids[3], strat.bids[4], strat.asks[0], strat.asks[1], strat.asks[2], strat.asks[3], strat.asks[4], strat.bid_sizes[0], strat.bid_sizes[1], strat.bid_sizes[2], strat.bid_sizes[3], strat.bid_sizes[4], strat.ask_sizes[0], strat.ask_sizes[1], strat.ask_sizes[2], strat.ask_sizes[3], strat.ask_sizes[4], strat.last_trade, strat.last_trade_size, strat.volume, strat.turnover, strat.open_interest, _, time_sec, time_usec, strat.is_initialized, res = struct.unpack(self.strat_fmt, content)
    strat.time = int(time_sec) + float("0."+str(time_usec))
    return strat.Filter()

  def read_border(self, i):
    if i >= self.order_struct_size:
      print("oversize for order %i > %i" %(i, self.order_struct_size))
      return Order()
    content = self.order_content[i*self.order_size:(i+1)*self.order_size]
    order = Order()
    #print(struct.unpack(self.order_fmt, content))
    shot_sec, shot_usec, send_sec, send_usec = 0,0,0,0
    shot_sec, shot_usec, send_sec, send_usec, order.ticker, order.price,order.size, order.traded_size, order.side, order.order_ref, order.action, order.status, order.offset, order.tbd = struct.unpack(self.order_fmt, content)
    order.shot_time = int(shot_sec) + float("0."+str(shot_usec))
    order.send_time = int(send_sec) + float("0."+str(send_usec))
    return order.Filter()

  def GetValidDate(self, path = '/running/'):
    self.valid_path_list = []
    date_list = os.listdir(path)
    for dl in date_list:
      data_file = path+dl+'/'+'data_binary.dat'
      if not os.path.exists(data_file):
        continue
      if os.path.getsize(data_file) < 100000:  # size < 100M will be ignored
        continue
      self.valid_path_list.append(data_file)
    self.valid_path_list = sorted(self.valid_path_list)
    return self.valid_path_list

  def LoadOneDF(self, file_path):
    if not os.path.exists(file_path):
      print('load df failed: %s not existed' % (file_path))
      shot = MarketSnapshot()
      return pd.DataFrame(columns = shot.get_columns())
    df = pd.read_csv(file_path)
    shot = MarketSnapshot()
    df.columns = shot.get_columns()
    return df

  def Labelling(self, df, future_ws = 50, win = 1, loss = 1, cut_tail = False):
    # snapshot[50] = slipwinmax()[100]
    if len(df) == 0:
      print('labelling passed empty df in!')
      return df
    df = df[df['asks[0]'] - df['bids[0]'] > 0]
    min_move = (df['asks[0]'] - df['bids[0]']).min()
    self.label_map = {'Flat': 0, 'Buy': 1, 'Sell': 2}
    if 'label' in df.columns:
      print('df has labelled!')
      return df
    ask_min = list(map(lambda x: -x, GetSlipWindowMax((df['asks[0]']*-1).tolist(), future_ws)))
    bid_max = GetSlipWindowMax(df['bids[0]'].tolist(), future_ws)
    #df = df[:-future_ws]
    df = df.drop([i for i in range(len(df)-future_ws, len(df))])
    ask_min = ask_min[future_ws:]
    bid_max = bid_max[future_ws:]
    #print('len(df)=%d, len(ask_min)=%d' % (len(df), len(bid_max)))
    df.insert(len(df.columns), 'label', 0)
    df.loc[df['bids[0]']-ask_min >= win*min_move, 'label'] = self.label_map['Sell']
    df.loc[bid_max-df['asks[0]'] >= win*min_move, 'label'] = self.label_map['Buy']
    '''
    print('ask_min:%s'%(str(ask_min[0:10])))
    print('bid_max:%s'%(str(bid_max[0:10])))
    print('ask:%s'%(str(df[0:10]['asks[0]'].tolist())))
    print('bid:%s'%(str(df[0:10]['bids[0]'].tolist())))
    print('sell profit:%s'%(str((df['bids[0]']-ask_min)[0:10].tolist())))
    print('buy  profit:%s'%(str((bid_max-df['asks[0]'])[0:10].tolist())))
    print('label:%s'%(str(df[:10]['label'].tolist())))
    #sys.exit(1)
    '''
    if cut_tail:
      df.loc[[i for i in range(500)], 'label'] = self.label_map['Flat']
      df.loc[[i for i in range(len(df)-500, len(df))], 'label'] = self.label_map['Flat']
    #print('after labelling: %.1f%% are buy, %.1f%% are sell, %.1f%% are flat' %((df['label']==self.label_map['Buy']).mean()*100, (df['label']==self.label_map['Sell']).mean()*100, (df['label']==self.label_map['Flat']).mean()*100))
    return df

if __name__=='__main__':
  r=Reader()
  date = '2020-03-19'
  order_file = '/running/'+date+'/order.dat'
  r.load_order_file(order_file)
  latency = {}
  for i in range(r.get_ordersize()):
    o = r.read_border(i)
    if o.action != 1:
      continue
    if o.shot_time < 100:
      continue
    o.Show()
    if o.ticker not in latency:
      latency[o.ticker] = []
    latency[o.ticker].append(o.send_time - o.shot_time)
  for i in latency:
    print("%s:%lfs" %(i, np.mean(latency[i])))
