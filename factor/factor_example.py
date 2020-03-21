from base_factor import *
class B(BaseFactor):
  def MACD(self, df, short_period, long_period, dea_period):
    ema_long = df['mid'].ewm(span=long_period, adjust=False).mean()
    ema_short = df['mid'].ewm(span=short_period, adjust=False).mean()
    diff = ema_long - ema_short
    dea = diff.ewm(span=dea_period, adjust=False).mean()
    bar = 2*(diff - dea)
    return bar

  def cal(self, df):
    short_period = [5, 10, 15]
    long_period = [2*i for i in short_period]
    dea_period = [6, 12, 20]
    return {"macd%d" %(short_period[i]): self.MACD(df, short_period[i], long_period[i], dea_period[i]) for i in range(len(short_period))}

if __name__ == '__main__':
  b = B(mode = 'tune', show_raw=False)
  b.run('2020-02-27', '2020-03-02', ['ni8888', 'cu8888', 'rb8888'])
