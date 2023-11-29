from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np

import requests
import config
import pandas as pd

client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
account = dict(client.get_account())


# for k,v in account.items():
#     print(f"{k:30}{v}")

def market_buy(symbol, qty):
    order_details = MarketOrderRequest(
        symbol= symbol,
        qty = qty,
        side = OrderSide.BUY,
        time_in_force = TimeInForce.DAY
    )
    order = client.submit_order(order_data= order_details)
    return order

def market_sell(symbol, qty):
    order_details = MarketOrderRequest(
        symbol= symbol,
        qty = qty,
        side = OrderSide.SELL,
        time_in_force = TimeInForce.DAY
    )
    order = client.submit_order(order_data= order_details)
    return order


# trades = TradingStream(config.API_KEY, config.SECRET_KEY, paper=True)
# async def trade_status(data):
#     print(data)
#
# trades.subscribe_trade_updates(trade_status)
# trades.run()

def show_positions():
    assets = [asset for asset in client.get_all_positions()]
    positions = [(asset.symbol, asset.qty, asset.current_price) for asset in assets]
    print("Postions")
    print(f"{'Symbol':9}{'Qty':>4}{'Value':>15}")
    print("-" * 28)
    for position in positions:
        print(f"{position[0]:9}{position[1]:>4}{float(position[1]) * float(position[2]):>15.2f}")
              
def close_all_positions():
    client.close_all_positions(cancel_orders=True)

def get_historical_data(symbol, start, end, limit, timeframe):
    api_key = config.API_KEY
    secret_key = config.SECRET_KEY
    headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    params={"start": start, "end": end, "limit": limit, "timeframe": timeframe}
    x = requests.get(config.data_url + f"/stocks/{symbol}/bars", headers=headers, params=params)
    print(x.status_code)
    return x.json()

def load_top_50():
    # get s&p 500 tickers
    snp500 = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    tickers = snp500['Symbol']
    top_50 = pd.DataFrame({'Ticker': tickers})
    top_50['market_cap'] = top_50['Ticker'].apply(top_50_helper)
    top_50 = top_50.sort_values(by='market_cap', ascending=False).head(50)
    top_50.to_csv('top_50_companies.csv', index=False)
    # snp500['market_cap'] = snp500['Symbol'].apply(top_50_helper)

def top_50_helper(ticker):
    api_key = config.API_KEY
    secret_key = config.SECRET_KEY
    headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    params={"start": "2023-11-01", "end": "2023-11-02", "limit": 1, "timeframe": '1Day'}
    r = requests.get(config.data_url + f"/stocks/{ticker}/bars", headers=headers, params=params).json()
    if 'bars' not in r:
        return 0
    market_cap = r['bars'][0]['c'] * r['bars'][0]['v']
    return market_cap

def get_clustering_data():
    sp500_list = pd.read_csv('top_50_companies.csv')
    sp500_historical = {}
    api_key = config.API_KEY
    secret_key = config.SECRET_KEY
    headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    params={"start": "2023-01-01", "limit": 400, "timeframe": '1Day'}
    for ticker in sp500_list['Ticker']:
        r = requests.get(config.data_url + f"/stocks/{ticker}/bars", headers=headers, params=params).json()
        df = pd.DataFrame(r['bars'])
        df['date'] = pd.to_datetime(df['t'])
        df['ticker'] = ticker
        sp500_historical[ticker] = df
    data = pd.concat(sp500_historical)
    data.reset_index(drop=True, inplace=True)
    data = data.pivot(index='date', columns='ticker', values = 'c')
    data.head(5)
    data.to_csv('S&P500_stock_data')

def clustering():
    # EDA
    data = pd.read_csv('S&P500_stock_data')
    pd.set_option('display.precision', 3)
    # print(data.describe().T.head(10))

    print(data.isnull().values.any())
    data = data.fillna(method='ffill')
    #Calculate returns and create a data frame
    returns = data.pct_change().mean()*266
    returns = pd.DataFrame(returns)
    returns.columns = ['returns']

    #Calculate the volatility
    returns['volatility'] = data.pct_change().std()*np.sqrt(266)

    data = returns
    data.head()
    



    


# amzn_json = get_historical_data("AMZN", "2023-03-01", "2023-03-02", 100, "1Min")

# # spy_json = get_historical_data("AMZN", "2021-01-01", "2021-01-02", 100, "1Min")
# amzn_df = pd.DataFrame(amzn_json['bars'])
# amzn_df = amzn_df.reindex(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n'])
# print(amzn_df)

if __name__ == "__main__":
    # load_top_50()                 # Load in this first if top_50_companies.csv is not found
    # get_clustering_data()         # Load in this first if S&P500_stock_data is not found
    clustering()