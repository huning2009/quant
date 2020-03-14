from Trader import *
from Dater import *
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from market_snapshot import *

shot=MarketSnapshot()
class BaseFactor:
  def __init__(self):
    self.f_value = []
    self.tr = Trader()

  def run(self, start_date, end_date, ticker):
    self.m = self.LoadData(start_date, end_date, ticker)
    self.CalFactor()
    self.PlotFactor()
    self.TestPnl()
    #self.PlotSignal()

  def PlotSignal(self):
    df_list = [i[1] for i in sorted(self.m.items(), key=lambda x:x[0])]
    if len(df_list) < 1:
      print('empty df')
      return
    df = df_list[0]
    start = time.time()
    for i in range(1, len(df_list)):
      df = pd.merge(df, df_list[i], how='outer')
    print('finished merge used %lfs' % (time.time() - start))
    plt.plot(df['mid'], label='mid', alpha=0.3)
    buy = df[df['signal']==1]
    plt.scatter(x=buy.index.tolist(), y=buy['mid'].tolist(), marker='.', s=[4]*len(buy), c='red', label='buy')
    sell = df[df['signal']==-1]
    plt.scatter(x=sell.index.tolist(), y=sell['mid'].tolist(), marker='.', s=[4]*len(sell), c='green', label='sell')
    plt.title('factor percentile signal')
    plt.legend()
    plt.grid()
    plt.show()

  def TestPnl(self, up_bound=0.9, down_bound=0.1, hold_one = True):
    for k in self.m:
      df = self.m[k]
      up = df['factor'].quantile(up_bound)
      down = df['factor'].quantile(down_bound)
      df['money'] = np.where(df['factor'] > up, df['mid'], 0.0)
      df['money'] = np.where(df['factor'] < down, -df['mid'], df['money'])
      signal_list = []
      last = 0.0
      for i, v in enumerate(df['money'].tolist()):
        if abs(v) < 1:
          signal_list.append(0)
          continue
        signal = (1 if v > 0 else -1)
        signal_list.append(signal)
        if hold_one:
          if last * v <= 0:
            self.tr.RegisterOneTrade('ni8888', signal, abs(v))
            last = v
        else:
            self.tr.RegisterOneTrade('ni8888', signal, abs(v))
      df['signal'] = signal_list
    self.tr.Summary()
    self.tr.PlotStratPnl(show=True)

  def PlotFactor(self):
    plt.plot(self.f_value, label='factor value')
    plt.title('factor value curve')
    plt.legend()
    plt.grid()
    plt.show()

  def CalFactor(self):
    for k in sorted(self.m.keys()):
      df = self.m[k]
      df['factor'] = self.cal(df)
      self.f_value += df['factor'].tolist()

  @abstractmethod
  def cal(self, df):
    pass

  def LoadData(self, start_date, end_date, ticker):
    dl = dateRange(start_date, end_date)
    m={}
    for date in dl:
      path = '/root/' + date + '/' + ticker + '.csv'
      if os.path.exists(path):
        m[date] = self.ReadData(date, ticker)
      else:
        print("%s not existed!" % (path))
    return m
  
  def ReadData(self, date, ticker):
    path = '/root/'+date+'/'+ticker+'.csv'
    df = pd.read_csv(path, header=None)
    df.columns = shot.get_columns()
    df['mid'] = (df['asks[0]'] + df['bids[0]']) / 2
    df['return1'] = df['mid'].diff(1).fillna(0.0)/df['mid']
    return df

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
    return bar

if __name__ == '__main__':
  a = A()
  a.run('2020-02-27', '2020-03-02', 'ni8888')
