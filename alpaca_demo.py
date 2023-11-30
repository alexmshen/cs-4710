from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
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
    top_50 = top_50.sort_values(by='market_cap', ascending=False).head(250)
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
    print("loaded clustering data")

def get_training_data():
    # EDA
    data = pd.read_csv('S&P500_stock_data')
    data.set_index('date', inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    pd.set_option('display.precision', 3)

    returns = data.pct_change().mean()*266
    returns = pd.DataFrame(returns)
    returns.columns = ['returns']


    #Calculate the volatility
    returns['volatility'] = data.pct_change().std()*np.sqrt(266)
    data = returns

    #Prepare the scaler
    scale = StandardScaler().fit(data)

    #Fit the scaler
    scaled_data = pd.DataFrame(scale.fit_transform(data),columns = data.columns, index = data.index)
    X = scaled_data
    print("loaded training data")
    return X

def plot_dendogram(X):
    plt.figure(figsize=(15, 10))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(X, method='ward'))
    plt.axhline(y=3, color='purple', linestyle='--')
    plt.show()
    clusters = 5
    hc = AgglomerativeClustering(n_clusters= clusters, affinity='euclidean', linkage='ward')
    labels = hc.fit_predict(X)

    #Plot the results
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow')
    ax.set_title('Hierarchical Clustering Results')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.show()

def k_means_eda(X):
    K = range(1,15)
    distortions = []

    #Fit the method
    for k in K:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    #Plot the results
    fig = plt.figure(figsize= (15,5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.show()

def k_means(X):
    c = 5
    #Fit the model
    k_means = KMeans(n_clusters=c)
    k_means.fit(X)
    prediction = k_means.predict(X)

    #Plot the results
    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize = (18,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c=k_means.labels_, cmap="rainbow", label = X.index)
    ax.set_title('k-Means Cluster Analysis Results')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=10)
    # plt.show()
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    return clustered_series

def find_coint_pairs(data, significance=0.05):
    n = data.shape[1]    
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i+1, n):
            S1 = data[keys[i]]            
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

def cointegration(clustered_series, data):
    cluster_dict = {}
    cluster_size_limit = 1000
    counts = clustered_series.value_counts()
    ticker_count = counts[(counts>1) & (counts<=cluster_size_limit)]

    for i, clust in enumerate(ticker_count.index):
        tickers = clustered_series[clustered_series == clust].index
        score_matrix, pvalue_matrix, pairs = find_coint_pairs(data[tickers])
        cluster_dict[clust] = {}
        cluster_dict[clust]['score_matrix'] = score_matrix
        cluster_dict[clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[clust]['pairs'] = pairs
        
    pairs = []   
    for cluster in cluster_dict.keys():
        pairs.extend(cluster_dict[cluster]['pairs'])
        
    print ("Number of pairs:", len(pairs))
    print ("In those pairs, we found %d unique tickers." % len(np.unique(pairs)))
    print(pairs)

    # display pairs
    stocks = np.unique(pairs)
    X_data = pd.DataFrame(index=X.index, data=X).T
    in_pairs_series = clustered_series.loc[stocks]
    stocks = list(np.unique(pairs))
    X_pairs = X_data.T.loc[stocks]
    X_pairs.head()

    # tnse algorithm
    X_tsne = TSNE(learning_rate=30, perplexity=3, random_state=42, n_jobs=-1).fit_transform(X_pairs)

    plt.figure(1, facecolor='white',figsize=(15,10))
    plt.clf()
    plt.axis('off')
    for pair in pairs:
        ticker1 = pair[0]
        loc1 = X_pairs.index.get_loc(pair[0])
        x1, y1 = X_tsne[loc1, :]
        ticker2 = pair[0]
        loc2 = X_pairs.index.get_loc(pair[1])
        x2, y2 = X_tsne[loc2, :]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='b');
        
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=215, alpha=0.8, c=in_pairs_series.values, cmap=cm.Paired)
    plt.title('TSNE Visualization of Pairs'); 

    # Join pairs by x and y
    for x,y,name in zip(X_tsne[:,0],X_tsne[:,1],X_pairs.index):

        label = name

        plt.annotate(label,
                    (x,y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
        
    plt.show()

    

if __name__ == "__main__":
    # load_top_50()                 # Load in this first if top_50_companies.csv is not found
    # get_clustering_data()         # Load in this first if S&P500_stock_data is not found
    X = get_training_data()
    # plot_dendogram(X)
    # k_means_eda(X)
    pairs_input = k_means(X)
    print(X.head())
    data = pd.read_csv('S&P500_stock_data')
    data.set_index('date', inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    cointegration(pairs_input, data)
