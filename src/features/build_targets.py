import os
import math
import pandas as pd

DATA_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\raw"
OUTPUT_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\interim"

def add_aboslute_percent_increase(data):
    data['%_change_adj_close_28'] = data['adj close'].pct_change(periods=28)
    data['%_change_adj_close_21'] = data['adj close'].pct_change(periods=21)
    data['%_change_adj_close_14'] = data['adj close'].pct_change(periods=14)
    data['%_change_adj_close_7'] = data['adj close'].pct_change(periods=7)
    return data

def add_percent_log_increase(data):
    data['log_adj_close'] = data['adj close'].apply(math.log10)
    data['%_change_log_adj_close_28'] = data['log_adj_close'].pct_change(periods=28)
    data['%_change_log_adj_close_21'] = data['log_adj_close'].pct_change(periods=21)
    data['%_change_log_adj_close_14'] = data['log_adj_close'].pct_change(periods=14)
    data['%_change_log_adj_close_7'] = data['log_adj_close'].pct_change(periods=7)
    return data

def encode_outcome_target(df, tgt_column, drawdown_pct, increase_pct, days):
    target_values = []
    #for each date in index get value 28 days ahead
    for date in df.index[:-days]:
        current_val = df.loc[date][tgt_column]
        min_future_val = current_val * (1+increase_pct)
        drawdown_amount = current_val * (1-drawdown_pct)
        #make sure values in between date and tgt date 
        #dont drop below drawdown pct target    
        end_date = date + pd.DateOffset(days=days)
        returns = df.loc[date:end_date][tgt_column]

        #identify if price drops below floor
        drawdown_indicator = int(any(map(lambda x: x<drawdown_amount, returns)))

        #identify if price goes above target
        return_indicator = int(any(map(lambda x: x>min_future_val, returns)))

        #if below floor return 0, if above target return 1, otherwise return 0
        if drawdown_indicator == 1:
            target = 0
        elif return_indicator == 0:
            target = 0
        else:
            target = 1
        target_values.append(target)
    #add null values for 
    for day in range(days):
        target_values.append(None)
    return target_values

def main():
    # get list of files
    data_files = [x for x in os.listdir(DATA_DIR) if ".gitkeep" not in x]

    #make outcome targets
    targets = [
        (.025, .05),
        (.05, .05),
        (.025, .1),
        (.05, .1),
        (.01, .03)
    ]

    time_horizons = [7,14,21,28]

    #for each data file make targets
    for file in data_files:
        print(f"Working on {file} data")
        data = pd.read_csv(os.path.join(DATA_DIR,file))
        data.columns = [x.lower() for x in data.columns]
        data['date'] = pd.to_datetime(data.date)
        data = data.set_index('date')

        #add return and log return targets
        data = add_aboslute_percent_increase(data)
        data = add_percent_log_increase(data)

        #add outcomes based targets
        for horizon in time_horizons:
            print(f"Generating {horizon} day targets")
            for risk, reward in targets:
                print(f"Creating {risk} drawdown {reward} increase target")
                target_title = f'{reward}%_higher_{risk}%_drawdown_{horizon}'
                data[target_title] = encode_outcome_target(df=data, 
                                                           tgt_column='adj close',
                                                           drawdown_pct=risk,
                                                           increase_pct=reward,
                                                           days=horizon)
        #drop ohlc columns
        to_drop = ['open','high','low','close','adj close','volume']
        data = data.drop(to_drop, axis=1)
        target_file_name = f"{file}_targets.csv"
        data.to_csv(os.path.join(OUTPUT_DIR,target_file_name))
        print(f"Complete work on {file}\n\n")

if __name__ == "__main__":
    main()