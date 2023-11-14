from config import ALPACA_CONFIG
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import numpy as np
import pandas as pd


class StatArb(Strategy):

    def initialize(self):
        signal = None
        start = "2022-01-01"

        self.signal = signal
        self.start = start
        self.sleeptime = "1D"
    # minute bars, make functions    

    def on_trading_iteration(self):
        stock1 = self.get_historical_prices("AMZN", 22, "day")
        stock2 = self.get_historical_prices("GOOG", 22, "day")
        amzn = stock1.df
        goog = stock2.df

        # Calculate the price ratio between the two stocks
        ratio = amzn['close'] / goog['close']

        # Calculate the z-score of the price ratio
        zscore = (ratio - np.mean(ratio)) / np.std(ratio)

        # Generate a signal when the z-score exceeds a certain threshold
        threshold = 0.5

        '''
        if zscore > threshold:
            symbol = "amzn"
            quantity = 200
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, quantity, "sell")
            self.submit_order(order)

            symbol = "goog"
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, quantity, "buy")
            self.submit_order(order)

        elif zscore < -threshold:
            symbol = "amzn"
            quantity = 200
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, quantity, "buy")
            self.submit_order(order)

            symbol = "goog"
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, quantity, "sell")
            self.submit_order(order)
        '''

        if goog['close'] - amzn['close'] >= 5.0:
            symbol1 = "GOOG"
            quantity = 200
            pos = self.get_position(symbol1)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol1, quantity, "sell")
            self.submit_order(order)

            symbol2 = "AMZN"
            pos = self.get_position(symbol2)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol2, quantity, "buy")
            self.submit_order(order)

        elif amzn['close'] - goog['close'] >= 5:
            symbol1 = "AMZN"
            quantity = 200
            pos = self.get_position(symbol1)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol1, quantity, "sell")
            self.submit_order(order)

            symbol2 = "GOOG"
            pos = self.get_position(symbol2)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol2, quantity, "buy")
            self.submit_order(order)
        
        elif abs(amzn['close'] - goog['close']) <= .1:
            symbol = "AMZN"
            quantity = 200
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()

            symbol = "GOOG"
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()

if __name__ == "__main__":
    trade = True
    if trade:
        broker = Alpaca(ALPACA_CONFIG)
        strategy = StatArb(broker=broker)
        bot = Trader()
        bot.add_strategy(strategy)
        bot.run_all()
    else:
        start = datetime(2022, 4, 15)
        end = datetime(2023, 4, 15)
        StatArb.backtest(
            YahooDataBacktesting,
            start,
            end
        )