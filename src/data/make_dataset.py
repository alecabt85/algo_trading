"""
This script downloads 5 year stock price data for all tickers
on the S&P500 and saves them as separate .csv files
"""
# -*- coding: utf-8 -*-
import yfinance as yf
import requests
from bs4 import BeautifulSoup as BS
import datetime
import os
import tqdm
import time
import random

#Set Data Dir
DATA_DIR = r"C:/Users/aburtnerabt/Documents/Continuing Education/Algo Trading/algo_trading/data/raw"

#TODO: get all tickers on S&P500
def get_tickers():
    """
    Function that retrieves all tickers for the S&P500 
    using the Wikipedia page
    """
    #get page with url
    url = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    page = requests.get(url, verify=False)

    #parse tickers with BS4
    soup = BS(page.text, 'html.parser')
    tickers = soup.select('#constituents tbody tr td:first-of-type ')
    tickers = [element.text.strip() for element in tickers]
    return tickers

    

#TODO: function to get historic price data for ticker at specified 
# time interval
def get_ticker_data(ticker):
    """
    Function that retrieves historic OHLC data for given ticker
    and time series type, will always default to max data
    """
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    tick = yf.download(ticker, period="5y")
    tick.to_csv(os.path.join(DATA_DIR, ticker))
    return

#TODO: store it in a local database or data file

if __name__ == "__main__":
    #get tickers
    tickers = get_tickers()
    
    #for each ticker get data, wait 5-10 seconds, then get next
    for symbol in tqdm.tqdm(tickers):
        get_ticker_data(symbol)
        time.sleep(random.randint(2, 6))
    
    