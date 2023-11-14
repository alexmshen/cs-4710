from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
import config

client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
account = dict(client.get_account())


for k,v in account.items():
    print(f"{k:30}{v}")

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

market_sell("AAPL", 10)