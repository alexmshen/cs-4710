from datetime import datetime
import backtrader as bt
import yfinance as yf

# Create a subclass of Strategy to define the indicators and logic

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=200   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

class Rsi(bt.Strategy):
    params = dict(
        period=14,
        upperband=70,
        lowerband=30
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.period)
        self.upperband = self.params.upperband
        self.lowerband = self.params.lowerband

    def next(self):
        if not self.position:
            if self.rsi < self.lowerband:
                self.buy()
        else:
            if self.rsi > self.upperband:
                self.sell()

class StatArb(bt.Strategy):
    params = dict(
        period=20,
        threshold=2.0
    )

    def __init__(self):
        self.data1 = self.datas[0]
        self.data2 = self.datas[1]
        self.spread = self.data1.close - self.data2.close
        self.sma = bt.indicators.SMA(self.spread, period=self.params.period)
        self.zscore = (self.spread - self.sma) / bt.indicators.StdDev(self.spread, period=self.params.period)

    def next(self):
        if self.zscore[0] > self.params.threshold:
            self.sell(data=self.data1)
            self.buy(data=self.data2)
        elif self.zscore[0] < -self.params.threshold:
            self.sell(data=self.data2)
            self.buy(data=self.data1)

cerebro = bt.Cerebro()

data1 = bt.feeds.PandasData(dataname=yf.download('COKE', '2020-01-01', '2023-11-09'))
data2 = bt.feeds.PandasData(dataname=yf.download('PEP', '2020-01-01', '2023-11-09'))

cerebro.adddata(data1)
cerebro.adddata(data2)

cerebro.addstrategy(StatArb)

# Create a data feed
'''
data = bt.feeds.PandasData(dataname=yf.download('TSLA', '2020-01-01', '2023-11-09'))
cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(SmaCross)  # Add the trading strategy
cerebro.addstrategy(Rsi)
'''
cerebro.run()  # run it all
cerebro.plot()  # and plot it with a single command