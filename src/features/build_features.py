import os
import tqdm
import pandas_ta as ta
import pandas as pd
from IPython.display import display

DATA_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\raw"
OUTPUT_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\interim"


def convert_to_ta(df):
    output = ta.DataFrame()
    for column in df.columns:
        output[column.lower()] = df[column]
    output.index = df.index
    return output

def add_momentum(df):
    df.ta.ao(append=True)
    df.ta.apo(append=True)
    df.ta.bias(append=True)
    df.ta.bop(append=True)
    df.ta.brar(append=True)
    df.ta.cci(append=True)
    df.ta.cfo(append=True)
    df.ta.cg(append=True)
    df.ta.cmo(append=True)
    df.ta.coppock(append=True)
    df.ta.cti(append=True)
    df.ta.dm(append=True)
    df.ta.er(append=True)
    df.ta.eri(append=True)
    df.ta.fisher(append=True)
    df.ta.inertia(append=True)
    df.ta.kdj(append=True)
    df.ta.kst(append=True)
    df.ta.macd(append=True)
    df.ta.mom(append=True)
    df.ta.pgo(append=True)
    df.ta.ppo(append=True)
    df.ta.psl(append=True)
    df.ta.pvo(append=True)
    df.ta.qqe(append=True, fillna=0)
    df.ta.roc(append=True)
    df.ta.rsi(append=True)
    df.ta.rsx(append=True)
    df.ta.rvgi(append=True)
    df.ta.stc(append=True)
    df.ta.slope(append=True)
    df.ta.smi(append=True)
    df.ta.squeeze(append=True)
    df.ta.squeeze_pro(append=True)
    df.ta.stoch(append=True)
    df.ta.trix(append=True)
    df.ta.tsi(append=True)
    df.ta.uo(append=True)
    df.ta.willr(append=True)
    return df

def add_overlap(df):
    df.ta.alma(append=True)
    df.ta.dema(append=True)
    df.ta.ema(append=True)
    df.ta.fwma(append=True)
    df.ta.hilo(append=True, fillna=0)
    df.ta.hl2(append=True)
    df.ta.hlc3(append=True)
    df.ta.hma(append=True)
    df.ta.hwma(append=True)
    df.ta.ichimoku(append=True, fill_method='ffill')
    df.ta.jma(append=True)
    df.ta.kama(append=True)
    df.ta.linreg(append=True)
    df.ta.mcgd(append=True)
    df.ta.midpoint(append=True)
    df.ta.midprice(append=True)
    df.ta.ohlc4(append=True)
    df.ta.pwma(append=True)
    df.ta.rma(append=True)
    df.ta.sinwma(append=True)
    df.ta.sma(append=True)
    df.ta.ssf(append=True)
    df.ta.supertrend(append=True, fillna=0)
    df.ta.swma(append=True)
    df.ta.t3(append=True)
    df.ta.tema(append=True)
    df.ta.trima(append=True)
    df.ta.vidya(append=True)
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    df.ta.vwap(append=True)
    df.ta.vwma(append=True)
    df.ta.wcp(append=True)
    df.ta.wma(append=True)
    df.ta.zlma(append=True)
    return df

def add_td_seq(df):
    out = df.ta.td_seq(show_all=True, fillna=0)
    out.index = df.index
    df = df.merge(out, left_index=True, right_index=True)
    return df

def add_candles(df):
    candles = df.ta.cdl_pattern(name='all')
    df = df.merge(candles, how='left', left_index=True, right_index=True)
    return df

def add_volatility(df):
    df.ta.aberration(append=True, fillna=0)
    df.ta.accbands(append=True, fillna=0)
    df.ta.atr(append=True, fillna=0)
    df.ta.bbands(append=True, fillna=0)
    df.ta.donchian(append=True, fillna=0)
    df.ta.hwc(append=True, fillna=0)
    df.ta.kc(append=True, fillna=0)
    df.ta.massi(append=True, fillna=0)
    df.ta.natr(append=True, fillna=0)
    df.ta.pdist(append=True, fillna=0)
    df.ta.rvi(append=True, fillna=0)
    df.ta.thermo(append=True, fillna=0)
    df.ta.true_range(append=True, fillna=0)
    df.ta.ui(append=True, fillna=0)
    return df

def feature_pipeline(file):
    #read in data
    data = pd.read_csv(file)

    #convert to pandas-ta df
    data = convert_to_ta(data)

    #add momentum, overlap, candles, vol 
    data = add_momentum(data)
    data = add_td_seq(data)
    data = add_overlap(data)
    data = add_candles(data)
    data = add_volatility(data)

    return data

if __name__ == "__main__":
    files = [x for x in os.listdir(DATA_DIR) if '.git' not in x]

    #for each file run the pipeline and save results to interim
    for file in tqdm.tqdm(files):
        data_file = os.path.join(DATA_DIR,file)
        data = feature_pipeline(data_file)
        feature_file_name = f"{file}_all_indicators_one_day.csv"
        data.to_csv(os.path.join(OUTPUT_DIR,feature_file_name))