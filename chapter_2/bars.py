import pandas as pd
import numpy as np
import os
import datetime

class BarSeries(object):

    def __init__(self, df, timecolumn='datetime'):
        self.df = df
        self.timecolumn = timecolumn

    def process_ohlc(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').ohlc()

    def process_volume(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').sum()

    def process_time(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').mean()

    def process_ticks(self, price_column='mid_price', volume_column='exe_q', time_column='time_diff', frequency='15Min'):

        ## Frequency examples
        # 1M  == 1 month
        # 1H == 1 hour
        # 1Min == 1 minute
        # 1S = 1 second
        # 1ms = 1 millisecond
        # 1U = 1 microsecond
        # 1N = 1 nanosecond

        ohlc_df = self.process_ohlc(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        time_df = self.process_time(time_column, frequency)
        ohlc_df['exe_q'] = volume_df
        ohlc_df['time_diff'] = time_df
        ohlc_df = ohlc_df.dropna()

        return ohlc_df

class TickBarSeries(BarSeries):

    def __init__(self, df, timecolumn='datetime', volume_column='exe_q', time_diff_column='time_diff'):
        self.volume_column = volume_column
        self.time_diff_column = time_diff_column
        super(TickBarSeries, self).__init__(df, timecolumn)

    def process_ohlc(self, column_name, frequency):

        data = []

        for i in range(frequency, len(self.df), frequency):
            sample = self.df.iloc[i - frequency:i]

            volume = sample[self.volume_column].values.sum()
            time_diff = sample[self.time_diff_column].values.mean()
            open = sample[column_name].values.tolist()[0]
            high = sample[column_name].values.max()
            low = sample[column_name].values.min()
            close = sample[column_name].values.tolist()[-1]
            time = sample.index.values[-1]

            data.append({
                self.timecolumn: time,
                'open': open,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'time_diff': time_diff,
            })

        data = pd.DataFrame(data).set_index(self.timecolumn)
        return data

    def process_ticks(self, price_column='mid_price', frequency='15Min'):
        ohlc_df = self.process_ohlc(price_column, frequency)
        return ohlc_df

class VolumeBarSeries(BarSeries):

    def __init__(self, df, timecolumn='datetime', volume_column='exe_q', time_diff_column='time_diff'):
        self.volume_column = volume_column
        self.time_diff_column = time_diff_column
        super(VolumeBarSeries, self).__init__(df, timecolumn)

    def process_ohlc(self, column_name, frequency):
        data = []
        buf = []
        time_diff_buff = []

        start_index = 0.
        volume_buf = 0.

        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            time_diff_value = self.df[self.time_diff_column].iloc[i]

            buf.append(pi)
            volume_buf += vi
            time_diff_buff.append(time_diff_value)

            if volume_buf >= frequency:

                open = buf[0]
                high = np.max(buf)
                low = np.min(buf)
                close = buf[-1]
                time_diff = np.mean(time_diff_buff)

                data.append({
                    self.timecolumn: di,
                    'open': open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume_buf,
                    'time_diff': time_diff
                })

                buf, time_diff_buff, volume_buf = [], [], 0.

        data = pd.DataFrame(data).set_index(self.timecolumn)
        return data

    def process_ticks(self, price_column = 'mid_price', volume_column='exe_q', frequency='15Min'):
        ohlc_df = self.process_ohlc(price_column, frequency)
        return ohlc_df

class DollarBarSeries(BarSeries):

    def __init__(self, df, timecolumn = 'datetime', volume_column='exe_q', time_diff_column='time_diff'):
        self.volume_column = volume_column
        self.time_diff_column = time_diff_column
        super(DollarBarSeries, self).__init__(df, timecolumn)

    def process_ohlc(self, column_name, frequency):

        data = []
        buf, vbuf, time_diff_buf = [], [], []
        start_index = 0.
        dollar_buf = 0.

        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            time_diff_value = self.df[self.time_diff_column].iloc[i]

            dvi = pi * vi
            buf.append(pi)
            vbuf.append(vi)
            dollar_buf += dvi
            time_diff_buf.append(time_diff_value)

            if dollar_buf >= frequency:

                open = buf[0]
                high = np.max(buf)
                low = np.min(buf)
                close = buf[-1]
                volume = np.sum(vbuf)
                time_diff = np.mean(time_diff_buf)

                data.append({
                    self.timecolumn: di,
                    'open': open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'dollar': dollar_buf,
                    'time_diff': time_diff,
                })

                buf, vbuf, time_diff_buf, dollar_buf = [], [], [], 0

        data = pd.DataFrame(data).set_index(self.timecolumn)
        return data

    def process_ticks(self, price_column='mid_price', volume_column='exe_q', frequency=10000):
        ohlc_df = self.process_ohlc(price_column, frequency)
        return ohlc_df

class ImbalanceTickBarSeries(BarSeries):

    def __init__(self, df, timecolumn='datetime', volume_column='exe_q'):
        self.volume_column = volume_column
        # self.time_diff_column = time_diff_column
        super(ImbalanceTickBarSeries, self).__init__(df, timecolumn)

    def get_bt(self, data):
        s = np.sign(np.diff(data))

        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]

        return s

    def get_theta_t(self, bt):
        return np.sum(bt)

    def ewma(self, data, window):

        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha

        scale = 1/alpha_rev
        n = data.shape[0]

        r = np.arange(n)
        scale_arr = scale ** r
        offset = data[0]*alpha_rev ** (r+1)
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out

    def process_ohlc(self, column_name, initial_T = 100, min_bar = 10, max_bar = 1000):
        init_bar = self.df[:initial_T][column_name].values.tolist()

        ts = [initial_T]
        bts = [bti for bti in self.get_bt(init_bar)]
        data = []

        time_diff_buf, buf_bar, vbuf, T = [], [], [], 0

        for i in range(initial_T, len(self.df)):

            di = self.df.index.values[i]

            buf_bar.append(self.df[column_name].iloc[i])
            bt = self.get_bt(buf_bar)
            theta_t = self.get_theta_t(bt)

            try:
                e_t = self.ewma(np.array(ts), initial_T / 10)[-1]
                e_bt = self.ewma(np.array(bts), initial_T)[-1]
            except:
                e_t = np.mean(ts)
                e_bt = np.mean(bts)
            finally:
                if np.isnan(e_bt):
                    e_bt = np.mean(bts[int(len(bts) * 0.9):])
                if np.isnan(e_t):
                    e_t = np.mean(ts[int(len(ts) * 0.9):])

            condition = np.abs(theta_t) >= e_t * np.abs(e_bt)

            if (condition or len(buf_bar) > max_bar) and len(buf_bar) >= min_bar:

                open = buf_bar[0]
                high = np.max(buf_bar)
                low = np.min(buf_bar)
                close = buf_bar[-1]
                volume = np.sum(vbuf)

                data.append({
                    self.timecolumn: di,
                    'open': open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                })

                ts.append(T)
                for b in bt:
                    bts.append(b)

                buf_bar = []
                vbuf = []
                T = 0.
            else:
                vbuf.append(self.df[self.volume_column].iloc[i])
                T += 1
        data = pd.DataFrame(data).set_index(self.timecolumn)
        return data

    def process_ticks(self, price_column = 'mid_price', volume_column='exe_q', init=100, min_bar = 10, max_bar = 1000):
        ohlc_df = self.process_ohlc(price_column, init, min_bar, max_bar)
        return ohlc_df
