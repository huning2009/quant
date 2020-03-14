from base_factor import *
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
