from Trader import *
from Dater import *
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from Plotor import *
from market_snapshot import *
import time

shot=MarketSnapshot()
class BaseFactor:
  def __init__(self, mode='tune', show_raw=False):  # tune->factor tune param test->multifactor test
    self.pt = Plotor()
    self.f_value = {}
    self.tr = Trader(show_raw=show_raw)
    self.return_list = []
    self.mode = mode

  def run(self, start_date, end_date, ticker):
    self.m = self.LoadData(start_date, end_date, ticker)
    fname = self.CalFactor()
    self.PlotFactor(factor_name=fname)
    self.TestPnl(factor_name=fname)
    #self.PlotSignal(factor_name=fname)
    start = time.time()
    self.CalIC(factor_name=fname)
    print('CALIC cost %lfs' % (time.time()-start))

  def InsertReturn(self, df, period_list):
    for pl in period_list:
      df['return'+str(pl)] = -df['mid'].diff(-pl)/df['mid']
      self.return_list.append('return'+str(pl))

  def ReadData(self, date, ticker):
    path = '/root/'+date+'/'+ticker+'.csv'
    df = pd.read_csv(path, header=None)
    df.columns = shot.get_columns()
    df['mid'] = (df['asks[0]'] + df['bids[0]']) / 2
    #df['return1'] = df['mid'].diff(1).fillna(0.0)/df['mid']
    self.InsertReturn(df, [3,5,10,20,50])
    return df

  def LoadData(self, start_date, end_date, ticker):
    dl = dateRange(start_date, end_date)
    m={}
    for t in ticker:
      for date in dl:
        path = '/root/' + date + '/' + t + '.csv'
        if os.path.exists(path):
          if t not in m:
            m[t] = {}
          m[t][date] = self.ReadData(date, t)
        else:
          print("%s not existed!" % (path))
    return m

  def CalFactor(self):
    mm ={}
    for t in self.m.keys():
      for k in sorted(self.m[t]):
        df = self.m[t][k]
        mm = self.cal(df)
        if len(set(mm.keys()) & set(df.columns.tolist())) != 0:
          print('duplicate columns name with factor name %s %s'%(str(mm.keys()), str(df.columns.tolist())))
          sys.exit(1)
        for fn in mm:
          df[fn] = mm[fn]
          if fn not in self.f_value:
            self.f_value[fn] = {}
          if fn not in self.f_value[fn]:
            self.f_value[fn][t] = []
          self.f_value[fn][t] += df[fn].tolist()
    return mm.keys()
 
  def PlotFactor(self, factor_name):
    for fn in factor_name:
      self.pt.MultiPlot(self.f_value[fn], fig_name='factor_value', show=True, prefix=fn)

  def TestPnl(self, factor_name, up_bound=0.9, down_bound=0.1, hold_one = True):
    if self.mode == 'tune':
      self.TunePnl(factor_name, up_bound, down_bound, hold_one)
    elif self.mode == 'test':
      self.TestPnl(factor_name, up_bound, down_bound, hold_one)
    else:
      print('unknown mode %s' % (self.mode))
      sys.exit(1)

  def TunePnl(self, factor_name, up_bound=0.9, down_bound=0.1, hold_one = True):
    for fn in factor_name:
      self.tr = Trader(prefix=fn)
      for t in self.m:
        for k in self.m[t]:
          df = self.m[t][k]
          up = df[fn].quantile(up_bound)
          down = df[fn].quantile(down_bound)
          df[fn+'_money'] = np.where(df[fn] > up, df['mid'], 0.0)
          df[fn+'_money'] = np.where(df[fn] < down, -df['mid'], df[fn+'_money'])
          signal_list = []
          last = 0.0
          for i, v in enumerate(df[fn+'_money'].tolist()):
            if abs(v) < 1:
              signal_list.append(0)
              continue
            signal = (1 if v > 0 else -1)
            if hold_one:
              if last * v <= 0:
                self.tr.RegisterOneTrade(t, signal, abs(v))
                last = v
                signal_list.append(signal)
              else:
                signal_list.append(0)
            else:
                self.tr.RegisterOneTrade(t, signal, abs(v))
                signal_list.append(signal)
          df[fn+'_signal'] = signal_list
      self.tr.Summary()
      self.tr.PlotStratPnl(show=True)

  def TestPnl(self, factor_name, up_bound=0.9, down_bound=0.1, hold_one = True):
    pnl = {}
    for fn in factor_name:
      self.tr = Trader(prefix=fn)
      for t in self.m:
        for k in self.m[t]:
          df = self.m[t][k]
          up = df[fn].quantile(up_bound)
          down = df[fn].quantile(down_bound)
          df[fn+'_money'] = np.where(df[fn] > up, df['mid'], 0.0)
          df[fn+'_money'] = np.where(df[fn] < down, -df['mid'], df[fn+'_money'])
          signal_list = []
          last = 0.0
          for i, v in enumerate(df[fn+'_money'].tolist()):
            if abs(v) < 1:
              signal_list.append(0)
              continue
            signal = (1 if v > 0 else -1)
            if hold_one:
              if last * v <= 0:
                self.tr.RegisterOneTrade(t, signal, abs(v))
                last = v
                signal_list.append(signal)
              else:
                signal_list.append(0)
            else:
                self.tr.RegisterOneTrade(t, signal, abs(v))
                signal_list.append(signal)
          df[fn+'_signal'] = signal_list
      self.tr.Summary()
      temp = self.tr.GetStratPnl(fn)
      if len(pnl) == 0:
        pnl.update(temp)
      else:
        for i in pnl:
          pnl[i].update(temp[i])
    self.pt.MultiPlot(pnl, show=True)

  def PlotSignal(self, factor_name):
    for fn in factor_name:
      tickers = self.m.keys()
      ksize = len(tickers)
      ncol, nrow = max(1, int(math.sqrt(ksize))), int(math.sqrt(ksize))+1
      fig,ax = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,6))
      fig.tight_layout()
      count = 0
      for t in self.m:
        df_list = [i[1] for i in sorted(self.m[t].items(), key=lambda x:x[0])]
        if len(df_list) < 1:
          print('empty df')
          return
        df = df_list[0]
        start = time.time()
        for i in range(1, len(df_list)):
          df = pd.merge(df, df_list[i], how='outer')
        print('finished merge used %lfs' % (time.time() - start))
        if count % (ncol*nrow) == 0 and count > 0:
            fig.savefig('%s@%s' %("signal", str(count)))
            fig,ax = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,6))
            print('finished @%d' %(count))
        fig.tight_layout()
        if ncol == 1:
          this_ax = ax[int(count/ncol)%nrow]
        else:
          this_ax = ax[int(count/ncol)%nrow, count%ncol]
        this_ax.plot(df['mid'], label='mid', alpha=0.3)
        buy = df[df[fn+'_signal']==1]
        this_ax.scatter(x=buy.index.tolist(), y=buy['mid'].tolist(), marker='.', s=[4]*len(buy), c='red', label='buy')
        sell = df[df[fn+'_signal']==-1]
        this_ax.scatter(x=sell.index.tolist(), y=sell['mid'].tolist(), marker='.', s=[4]*len(sell), c='black', label='sell')
        this_ax.set_title(fn+'_'+t)
        this_ax.legend()
        this_ax.grid()
        count += 1
      plt.show()


  @abstractmethod
  def cal(self, df):
    pass

  def save_data(self):
    #np.save(self.m, 'output.npy')
    print(self.m['ni8888']['2020-02-28'].columns)

  def CalIC(self, factor_name):
    ic_map = {}
    for t in self.m:
      if t not in ic_map:
        ic_map[t] = {}
      for r in self.return_list:
        if r not in ic_map[t]:
          ic_map[t][r] = {}
        for fn in factor_name:
          corr = []
          for d in self.m[t]:
            corr.append(abs(self.m[t][d][r].corr(self.m[t][d][fn])))
          ic_map[t][r][fn] = np.mean(corr)
    ic = {}
    for i in ic_map:
      ic[i] = pd.DataFrame(ic_map[i])
      print(ic[i])

  def CalIC2(self, factor_name):
    ic_map = {}
    for t in self.m:
      if t not in ic_map:
        ic_map[t] = {}
      for r in self.return_list:
        if r not in ic_map[t]:
          ic_map[t][r] = {}
        for fn in factor_name:
          corr = []
          for d in self.m[t]:
            corr.append(abs(self.m[t][d][r].corr(self.m[t][d][fn])))
          ic_map[t][r][fn] = np.mean(corr)
    ic = {}
    for i in ic_map:
      ic[i] = pd.DataFrame(ic_map[i])
      print(ic[i])

class A(BaseFactor):
  def cal(self, df):
    short_period = 10
    long_period = 20
    dea_period = 12
    ema_long = df['mid'].ewm(span=long_period, adjust=False).mean()
    ema_short = df['mid'].ewm(span=short_period, adjust=False).mean()
    diff = ema_long - ema_short
    dea = diff.ewm(span=dea_period, adjust=False).mean()
    bar = 2*(diff - dea)
    return {'macd': bar, 'midf':-df['mid']}

if __name__ == '__main__':
  a = A(mode='tune')
  a.run('2020-02-27', '2020-03-02', ['ni8888', 'zn8888', 'sn8888', 'cu8888', 'ag8888'])
  #a.save_data()
