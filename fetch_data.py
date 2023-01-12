
import pandas as pd
from datetime import datetime, timezone, timedelta
import calendar

pd.options.plotting.backend = "plotly"

from math import sqrt
import numpy as np


def WMA(s, period):
    return s.rolling(period).apply(lambda x: ((np.arange(period) + 1) * x).sum() / (np.arange(period) + 1).sum(),
                                   raw=True)


def HMA(s, period):
    return WMA(WMA(s, period // 2).multiply(2).sub(WMA(s, period)), int(np.sqrt(period)))


def get_klines_iter(symbol, interval, start, end=None, limit=1000):
    # start and end must be isoformat YYYY-MM-DD
    # We are using utc time zone

    # the maximum records is 1000 per each Binance API call

    df = pd.DataFrame()

    if start is None:
        print('start time must not be None')
        return
    start = calendar.timegm(datetime.fromisoformat(start).timetuple()) * 1000

    if end is None:
        dt = datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        end = int(utc_time.timestamp()) * 1000
        return
    else:
        end = calendar.timegm(datetime.fromisoformat(end).timetuple()) * 1000
    last_time = None

    while len(df) == 0 or (last_time is not None and last_time < end):
        url = 'https://data.binance.com/api/v3/klines?symbol=' + \
              symbol + '&interval=' + interval + '&limit=1000'
        if (len(df) == 0):
            url += '&startTime=' + str(start)
        else:
            url += '&startTime=' + str(last_time)

        url += '&endTime=' + str(end)
        df2 = pd.read_json(url)
        df2.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime',
                       'Quote asset volume', 'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore']
        dftmp = pd.DataFrame()
        dftmp = pd.concat([df2, dftmp], axis=0, ignore_index=True, keys=None)

        dftmp.Opentime = pd.to_datetime(dftmp.Opentime, unit='ms')
        dftmp['Date'] = dftmp.Opentime.dt.strftime("%d/%m/%Y")
        dftmp['Time'] = dftmp.Opentime.dt.strftime("%H:%M:%S")
        dftmp = dftmp.drop(['Quote asset volume', 'Closetime', 'Opentime',
                            'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore'], axis=1)
        column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        dftmp.reset_index(drop=True, inplace=True)
        dftmp = dftmp.reindex(columns=column_names)
        string_dt = str(dftmp['Date'][len(dftmp) - 1]) + 'T' + str(dftmp['Time'][len(dftmp) - 1]) + '.000Z'
        utc_last_time = datetime.strptime(string_dt, "%d/%m/%YT%H:%M:%S.%fZ")
        last_time = (utc_last_time - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
        df = pd.concat([df, dftmp], axis=0, ignore_index=True, keys=None)

    return df



def get_hull_new_dataframe(df):
    df_final = df.copy()

    df_final = df_final.drop_duplicates(subset='Datetime')
    df_final.index = range(0, len(df_final))
    df_final['Close_7'] = df_final.Close.rolling(4).mean()

    period = 3
    df_final['HMA'] = HMA(df_final.Close, period)
    df_final['Pattern'] = df_final['HMA'] > df_final['HMA'].shift(2)
    df_final['TREND'] = 'D'
    df_final.loc[df_final['Pattern'] == True, 'TREND'] = 'U'

    df_final['HULL_Buy_entry'] = False
    df_final['HULL_Buy_exit'] = False
    df_final['HULL_Sell_entry'] = False
    df_final['HULL_Sell_exit'] = False

    counter_buy = 0
    counter_sell = 0

    loop_b = False
    for row, idx in zip(df_final.to_dict(orient="records"), df_final.index.values):
        if (row['TREND'] == 'U') and loop_b == False:
            df_final.loc[idx, 'HULL_Buy_entry'] = True
            loop_b = True

        if row['TREND'] != 'U' and loop_b:
            df_final.loc[idx, 'HULL_Buy_exit'] = True
            loop_b = False

    loop_b = False
    for row, idx in zip(df_final.to_dict(orient="records"), df_final.index.values):

        if (row['TREND'] == 'D') and loop_b == False:
            df_final.loc[idx, 'HULL_Sell_entry'] = True
            loop_b = True

        if row['TREND'] != 'D' and loop_b:
            df_final.loc[idx, 'HULL_Sell_exit'] = True
            loop_b = False

    return df_final


import os

try:
    os.mkdir("results")
except:
    pass

from ta.volatility import AverageTrueRange
from ta.trend import ema_indicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator, RSIIndicator


def get_RSI(close, window=26, fillna=False):
    rsi_ = RSIIndicator(close, window, fillna=fillna)

    return rsi_.rsi()


import traceback


def get_SMA(close, high=26, low=12, fillna=False):
    long_ = SMAIndicator(close, high, fillna=fillna)
    short_ = SMAIndicator(close, low, fillna=fillna)

    return long_.sma_indicator(), short_.sma_indicator(), long_.sma_indicator() - short_.sma_indicator()


def get_EMA(close, high=26, low=12, fillna=False):
    long_ = EMAIndicator(close, high, fillna=fillna)
    short_ = EMAIndicator(close, low, fillna=fillna)

    return long_.ema_indicator(), short_.ema_indicator(), long_.ema_indicator() - short_.ema_indicator()


def get_MACD(close, fast=26, slow=12, smooth=9, fillna=False):
    macd = MACD(close, fast, slow, fillna=fillna)

    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def get_BBands(close, window=26, std=2, fillna=False):
    bbands = BollingerBands(close, window, std, fillna=fillna)

    return bbands.bollinger_hband(), bbands.bollinger_lband(), bbands.bollinger_hband() - close, bbands.bollinger_lband() - close


def get_Stoch_Osc(df_ta, window=14, smoothing=3, lower_cross=20, upper_cross=80, fill_na=False):
    stochs = StochasticOscillator(high=df_ta['High'], low=df_ta['Low'], close=df_ta['Close'], window=window,
                                  smooth_window=smoothing, fillna=False)
    # df_p = df_ta.ta.stoch(high='High', low='Low', k=14, d=3, append=True)

    stochastic = stochs.stoch()
    stochastic_sigal = stochs.stoch_signal()

    return stochastic, stochastic_sigal


def get_RSI(close, window=26, fillna=False):
    rsi_ = RSIIndicator(close, window, fillna=fillna)

    return rsi_.rsi()


def get_PCHANGE(close, period=26, fillna=False):
    return close.pct_change(periods=period) * 100


def get_more_indicataors(data):
    data = data.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'})

    data['RSI'] = get_RSI(data['Close'], window=14)

    data['BBANDS_high'], data['BBANDS_low'], data['BBUP_diff'], data['BBLW_diff'] = get_BBands(data['Close'], window=20,
                                                                                               std=2)

    data['SMA_long_200'], data['SMA_short_100'], data['SMA_diff'] = get_SMA(data['Close'], high=200, low=100)
    data['SMA_long_100'], data['SMA_short_50'], data['SMA_diff'] = get_SMA(data['Close'], high=100, low=50)
    data['SMA_long_50'], data['SMA_short_20'], data['SMA_diff'] = get_SMA(data['Close'], high=50, low=20)

    data['EMA_long_50'], data['EMA_short_21'], data['EMA_diff_50'] = get_EMA(data['Close'], high=50, low=21)
    data['EMA_long_26'], data['EMA_short_9'], data['EMA_diff'] = get_EMA(data['Close'], high=26, low=9)

    data['MACD'], data['MACD_signal'], data['MACD_diff'] = get_MACD(data['Close'], fast=26, slow=12)

    data['PCT_change'] = get_PCHANGE(data['Close'], period=15)
    data['STOCH_14'], data['STOCH_signal_14'] = get_Stoch_Osc(data, window=14, smoothing=3)
    data['STOCH_14'], data['STOCH_signal_14'] = get_Stoch_Osc(data, window=40, smoothing=3)
    # data['ATR_10'] = AverageTrueRange(df.High, df.Low , df.Close, window = c, fillna = False).average_true_range()
    data = data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'Volume'})

    return data


import pandas as pd


class Grid():

    # Initialization
    def __init__(self, df="None"):

        self.data = df

        # Account leverage
        self.leverage = 100

        # Equity increase for take profit
        self.take_profit = 0.0025

        # Forex standard currency lot (Units)
        self.total_cash = 10000

        # Number of grid levels
        self.gridlevels = 30

        # Base pip number for grid
        self.gridpipsize = 1
        self.current_direction = "None"

        self.long_grid_used = 0
        self.short_grid_used = 0

        # Grid direction for pair 1
        self.direction1 = 'LONG'
        self.direction2 = 'SHORT'

        self.pnldf = pd.DataFrame(
            columns=['date', 'pnl', 'total_pnl', 'grid_number', 'long_grid_used', 'short_grid_used'])
        self.pnllist = []
        self.total_pnl = 0

        self.n = 1
        # Grid creation flag
        self.create_grid = False

        self.orderlist = pd.DataFrame(
            columns=['datetime', 'size', 'price', 'type', 'status', 'executed_price', 'PNL', 'closed_price',
                     'closed_date'])

    def StopMarketOrder(self, size, price, types, date):

        self.orderlist.loc[len(self.orderlist)] = [date, size, price, types, 'queued', 0, 0, 0, 0]

    def run_data_on_strategy(self):

        for i in self.data.index.values:
            # if i%500 == 0:
            #   print("At iteration {} total PNL : {} , orderlist PNL : {}".format(str(i), str(self.total_pnl), self.orderlist.PNL.sum()))
            self.pnllist.append(self.total_pnl)
            self.OnData(self.data.loc[i], self.data.loc[i]['Datetime'])

    # On data operations
    def OnData(self, data, date):

        self.exectue_orders(data['Close'], date, data['HULL_Buy_exit'], data['HULL_Sell_exit'])
        # Check if new grid needs to be created
        if self.create_grid == False:
            print("*************** Creating Grid {} ****************".format(str(self.n)))
            self.price1 = data['Close']
            self.CreateGrid(self.direction1, self.price1, date)
            self.CreateGrid(self.direction2, self.price1, date)
            self.create_grid = True

    def find_return(self, in_amount, ins, outs):

        fees = 0.02
        amount = in_amount * self.leverage
        return (((1 / ins) - (1 / outs)) * amount * outs) - (amount / 100) * fees * 2

    def liquidate(self, date):

        self.total_pnl = self.total_pnl + self.orderlist.PNL.sum()
        self.pnldf.loc[len(self.pnldf)] = [date, self.orderlist.PNL.sum(), self.total_pnl, self.n, self.long_grid_used,
                                           self.short_grid_used]
        self.orderlist.to_csv("results/grid_{}_results.csv".format(str(self.n)))
        self.n = self.n + 1
        self.orderlist = pd.DataFrame(
            columns=['datetime', 'size', 'price', 'type', 'status', 'executed_price', 'PNL', 'closed_price',
                     'closed_date'])
        self.total_cash = 1000
        self.create_grid = False
        self.current_direction = "None"
        self.long_grid_used = 0
        self.short_grid_used = 0

    def exectue_orders(self, currentprice, date, buy_exit, sell_exit):

        if len(self.orderlist[self.orderlist['status'] == 'executed']) > 0:

            for i in self.orderlist[self.orderlist['status'] == 'executed'].index.values:

                if self.orderlist.loc[i]['type'] == 'long':
                    self.orderlist.loc[i, 'PNL'] = self.find_return(self.orderlist.loc[i]['size'],
                                                                    self.orderlist.loc[i]['executed_price'],
                                                                    currentprice)
                    self.orderlist.loc[i, 'closed_price'] = currentprice
                    self.orderlist.loc[i, 'closed_date'] = date

                else:
                    self.orderlist.loc[i, 'PNL'] = self.find_return(self.orderlist.loc[i]['size'], currentprice,
                                                                    self.orderlist.loc[i]['executed_price'])
                    self.orderlist.loc[i, 'closed_price'] = currentprice
                    self.orderlist.loc[i, 'closed_date'] = date

        if self.current_direction == 'long' and buy_exit:
            print("Closing Grid {} at PNL {}".format(str(self.n), str(self.orderlist.PNL.sum())))

            self.liquidate(date)

        # print(date,self.current_direction, sell_exit )
        if self.current_direction == 'short' and sell_exit:
            print("Closing Grid {} at PNL {}".format(str(self.n), str(self.orderlist.PNL.sum())))
            self.liquidate(date)

        if len(self.orderlist[self.orderlist['status'] == 'queued']) > 0:

            filtered_df_long = self.orderlist[
                (self.orderlist['type'] == 'long') & (self.orderlist['status'] == 'queued') & (
                            self.orderlist['price'] < currentprice)]
            filtered_df_short = self.orderlist[
                (self.orderlist['type'] == 'short') & (self.orderlist['status'] == 'queued') & (
                            self.orderlist['price'] > currentprice)]

            # print("CC PRICE :", currentprice)
            # print(filtered_df_long)

            for idx in filtered_df_long.index.values:
                self.current_direction = "long"
                self.orderlist.loc[idx, 'status'] = 'executed'
                self.orderlist.loc[idx, 'executed_price'] = self.orderlist.loc[idx, 'price']
                self.long_grid_used += 1

            for idx in filtered_df_short.index.values:
                self.current_direction = "short"
                self.orderlist.loc[idx, 'status'] = 'executed'
                self.orderlist.loc[idx, 'executed_price'] = self.orderlist.loc[idx, 'price']
                self.short_grid_used += 1

    # Grid creation
    def CreateGrid(self, direction, start_price, date):
        if direction == 'LONG':
            for i in range(1, self.gridlevels + 1):
                size = 100
                price = start_price + (start_price / 100 * 0.20 * i)
                # print(price, start_price)
                self.StopMarketOrder(size, price, 'long', date)

        elif direction == 'SHORT':
            for i in range(1, self.gridlevels + 1):
                size = 100
                price = start_price - (start_price / 100 * 0.20 * i)
                self.StopMarketOrder(size, price, 'short', date)

