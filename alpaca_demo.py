from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
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

# amzn_json = get_historical_data("AMZN", "2023-03-01", "2023-03-02", 100, "1Min")

# # spy_json = get_historical_data("AMZN", "2021-01-01", "2021-01-02", 100, "1Min")
# amzn_df = pd.DataFrame(amzn_json['bars'])
# amzn_df = amzn_df.reindex(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n'])
# print(amzn_df)
