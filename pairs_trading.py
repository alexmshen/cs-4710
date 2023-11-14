import threading
from time import sleep
 
import alpaca_trade_api as tradeapi
import pandas as pd
 
base_url = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'
API_KEY = "PK7IJNU5PF46X2MJCFAB"
SECRET_KEY = "WbIZu5kHkPUReBnwV3TOZZ7qAVs3vXHaP0qB8ZdC"

trade_taken = False
 
# instantiate REST API
api = tradeapi.REST(base_url=base_url, key_id=API_KEY, secret_key=SECRET_KEY, api_version='v2')
 
# init WebSocket
conn = tradeapi.Stream(
    base_url=base_url, key_id=API_KEY, secret_key=SECRET_KEY
)

def wait_for_market_open():
    clock = api.get_clock()
    if not clock.is_open:
        time_to_open = clock.next_open - clock.timestamp
        sleep(time_to_open.total_seconds())
    return clock

# define websocket callbacks
data_df = None
 

conn.run()
