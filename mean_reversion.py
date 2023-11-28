import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import seaborn as sns
import alpaca_demo

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

sns.set_style("darkgrid")

def create_pairs_dataframe(datadir, symbols):
    """
    Creates a pandas DataFrame containing the closing price
    of a pair of symbols based on CSV files containing a datetime
    stamp and OHLCV data.

    Parameters
    ----------
    datadir : `str`
        Directory location of CSV files containing OHLCV intraday data.
    symbols : `tup`
        Tuple containing ticker symbols as `str`.

    Returns
    -------
    pairs : `pd.DataFrame`
        A DataFrame containing Close price for SPY and IWM. Index is a 
        Datetime object.
    """
    # Open the individual CSV files and read into pandas DataFrames 
    # using the first column as an index and col_names as the headers
    
    aapl_json = alpaca_demo.get_historical_data("AAPL", "2023-03-01", "2023-03-10", 1000, "1Min")
    sym1 = pd.DataFrame(aapl_json['bars'])
    sym1 = sym1.reindex(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n'])
    sym1 = sym1.rename(columns={'t': 'datetime', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'na'})

    amzn_json = alpaca_demo.get_historical_data("AMZN", "2023-03-01", "2023-03-10", 1000, "1Min")
    sym2 = pd.DataFrame(amzn_json['bars'])
    sym2 = sym2.reindex(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n'])
    sym2 = sym2.rename(columns={'t': 'datetime', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'na'})

    # print("Importing CSV data...")
    # col_names = ['datetime','open','high','low','close', 'volume', 'na']
    # sym1 = pd.read_csv(
    #     os.path.join(datadir, '%s.csv' % symbols[0]),
    #     header=0,
    #     index_col=0,
    #     names=col_names
    # )
    # sym2 = pd.read_csv(
    #     os.path.join(datadir, '%s.csv' % symbols[1]),
    #     header=0,
    #     index_col=0,
    #     names=col_names
    # )

    # Create a pandas DataFrame with the close prices of each symbol
    # correctly aligned and dropping missing entries
    print("Constructing dual matrix for %s and %s..." % symbols)
    pairs = pd.DataFrame(index=sym1.index)
    pairs['%s_close' % symbols[0].lower()] = sym1['close']
    pairs['%s_close' % symbols[1].lower()] = sym2['close']
    pairs.index = pd.to_datetime(pairs.index)
    pairs = pairs.dropna()
    return pairs

def calculate_spread_zscore(pairs, symbols, lookback=100):
    """
    Creates a hedge ratio between the two symbols by calculating
    a rolling linear regression with a defined lookback period. This
    is then used to create a z-score of the 'spread' between the two
    symbols based on a linear combination of the two.

    Parameters
    ----------
    pairs : `pd.DataFrame`
        A DataFrame containing Close price for SPY and IWM. Index is a 
        Datetime object.
    symbols : `tup`
        Tuple containing ticker symbols as `str`.
    lookback : `int`, optional (default: 100)
        Lookback preiod for rolling linear regression.

    Returns
    -------
    pairs : 'pd.DataFrame'
        Updated DataFrame containing the spread and z score between
        the two symbols based on the rolling linear regression.    
    """

    # Use the statsmodels Rolling Ordinary Least Squares method to fit
    # a rolling linear regression between the two closing price time series
    print("Fitting the rolling Linear Regression...")

    model = RollingOLS(
        endog=pairs['%s_close' % symbols[0].lower()],
        exog=sm.add_constant(pairs['%s_close' % symbols[1].lower()]),
        window=lookback
    )
    rres = model.fit()
    params = rres.params.copy()
    
    # Construct the hedge ratio and eliminate the first 
    # lookback-length empty/NaN period
    pairs['hedge_ratio'] = params['amzn_close']
    pairs.dropna(inplace=True)

    # Create the spread and then a z-score of the spread
    print("Creating the spread/zscore columns...")
    pairs['spread'] = (
        pairs['aapl_close'] - pairs['hedge_ratio']*pairs['amzn_close']
    )
    pairs['zscore'] = (
        pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread']
    )

    return pairs

def create_long_short_market_signals(
        pairs, symbols, z_entry_threshold, z_exit_threshold
    ):
    """
    Create the entry/exit signals based on the exceeding of z_entry_threshold
    for entering a position and falling below z_exit_threshold for exiting
    a position.

    Parameters
    ----------
    pairs : `pd.DataFrame`
        Updated DataFrame containing the close price, spread and z score
        between the two symbols.
    symbols : `tup`
        Tuple containing ticker symbols as `str`.
    z_entry_threshold : `float`, optional (default:2.0)
        Z Score threshold for market entry. 
    z_exit_threshold : `float`, optional (default:1.0)
        Z Score threshold for market exit.

    Returns
    -------
    pairs : `pd.DataFrame`
        Updated DataFrame containing long, short and exit signals.
    """

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold by still greater
    # than z_exit_threshold, and vice versa for shorts.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    
    # These variables track whether to be long or short while
    # iterating through the bars
    long_market = 0
    short_market = 0


    # Calculates when to actually be "in" the market, i.e. to have a
    # long or short position, as well as when not to be.
    # Since this is using iterrows to loop over a dataframe, it will
    # be significantly less efficient than a vectorised operation,
    # i.e. slow!
    print("Calculating when to be in the market (long and short)...")
    for i, b in enumerate(pairs.iterrows()):
        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1            
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.loc[b[0], 'long_market'] = long_market
        pairs.loc[b[0], 'short_market'] = short_market
    return pairs

def create_portfolio_returns(pairs, symbols):
    """
    Creates a portfolio pandas DataFrame which keeps track of
    the account equity and ultimately generates an equity curve.
    This can be used to generate drawdown and risk/reward ratios.
    
    Parameters
    ----------
    pairs : `pd.DataFrame`
        Updated DataFrame containing the close price, spread and z score
        between the two symbols and the long, short and exit signals.
    symbols : `tup`
        Tuple containing ticker symbols as `str`.

    Returns
    -------
    portfolio : 'pd.DataFrame'
        A DataFrame with datetime index from the pairs DataFrame, positions,
        total market value and returns.
    """
    
    # Convenience variables for symbols
    sym1 = symbols[0].lower()
    sym2 = symbols[1].lower()

    # Construct the portfolio object with positions information
    # Note the minuses to keep track of shorts!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs['%s_close' % sym1] * portfolio['positions']
    portfolio[sym2] = pairs['%s_close' % sym2] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    # Construct a percentage returns stream and eliminate all 
    # of the NaN and -inf/+inf cells
    print("Constructing the equity curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio

if __name__ == "__main__":
    datadir = '/Users/alexandershen/Desktop/trdalgo/cs-4710/'  # Change this to reflect your data path!
    symbols = ('AAPL', 'AMZN')

    lookbacks = range(50, 400, 10)
    returns = []

    # Adjust lookback period from 50 to 200 in increments
    # of 10 in order to produce sensitivities
    for lb in lookbacks: 
        print("Calculating lookback=%s..." % lb)
        pairs = create_pairs_dataframe(datadir, symbols)
        pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
        pairs = create_long_short_market_signals(
            pairs, symbols, z_entry_threshold=1, z_exit_threshold=.5
        )
        portfolio = create_portfolio_returns(pairs, symbols)
        returns.append(portfolio.iloc[-1]['returns'])

    print("Plot the lookback-performance scatterchart...")
    plt.plot(lookbacks, returns, '-o')
    plt.show()

    # This is still within the main function
    print("Plotting the performance charts...")
    fig = plt.figure()

    ax1 = fig.add_subplot(311,  ylabel='%s growth (%%)' % symbols[0])
    (pairs['%s_close' % symbols[0].lower()].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)

    ax2 = fig.add_subplot(312, ylabel='%s growth (%%)' % symbols[1])
    (pairs['%s_close' % symbols[1].lower()].pct_change()+1.0).cumprod().plot(ax=ax2, color='b', lw=2.)

    ax3 = fig.add_subplot(313, ylabel='Portfolio value growth (%%)')
    portfolio['returns'].plot(ax=ax3, lw=2.)

    plt.show()
