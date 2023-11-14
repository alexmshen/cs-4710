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
api = tradeapi.REST(base_url=base_url, APCA_API_KEY_ID=API_KEY, APCA_API_SECRET_KEY=SECRET_KEY, api_version='v2')
 
# init WebSocket
conn = tradeapi.stream2.StreamConn(
    base_url=base_url, data_url=data_url, data_stream='alpacadatav1'
)

def wait_for_market_open():
    clock = api.get_clock()
    if not clock.is_open:
        time_to_open = clock.next_open - clock.timestamp
        sleep(time_to_open.total_seconds())
    return clock

# define websocket callbacks
data_df = None
 
@conn.on(r'^T.ENZL$')
async def on_second_bars_EWN(conn, channel, bar):
    if data_df is not None:
        data_df.enzl[-1] = bar.price
 
 
@conn.on(r'^T.EWA$')
async def on_second_bars_ENZL(conn, channel, bar):
    if data_df is not None:
        data_df.ewa[-1] = bar.price
 
 
streams = ['T.ENZL', 'T.EWA']
ws_thread = threading.Thread(target=conn.run, daemon=True, args=(streams,))
ws_thread.start()