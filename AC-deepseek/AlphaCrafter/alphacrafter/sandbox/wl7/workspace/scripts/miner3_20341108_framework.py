"""miner_3 factor validation framework (2034-11-08).
Builds cross-asset close panel through visible_through date, computes factor
values and forward returns, then reports per-horizon IC/ICIR/coverage etc.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

with open('../persistent/date.json') as f:
    date_info = json.load(f)
VIS = date_info['visible_through']

def load_close(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.to_datetime(VIS)]
    s = df.set_index('date')['close'].astype(float)
    s = s[~s.index.duplicated(keep='first')]
    return s

def load_series(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.to_datetime(VIS)]
    df = df.set_index('date')
    df = df[~df.index.duplicated(keep='first')]
    return df

def build_panel():
    closes = {}
    volumes = {}
    for a in ASSETS:
        s = load_close(a)
        closes[a] = s
        df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.to_datetime(VIS)]
        df = df.set_index('date')
        df = df[