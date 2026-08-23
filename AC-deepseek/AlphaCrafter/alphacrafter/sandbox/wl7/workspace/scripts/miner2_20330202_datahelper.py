import os, json
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "persistent")
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd

def load_watchlist():
    acct = get_account_dict()
    return list(acct.get("watch_list", []))

UNIVERSE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
SIGNALS = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load_closes(days=1800):
    out = {}
    for s in UNIVERSE:
        df = get_index_daily_data(symbol=s, days=days)
        if df is None or len(df)==0:
            df = get_stock_daily_data(symbol=s, days=days)
        out[s] = df
    return out

def load_prices(days=1800):
    out = {}
    for s in UNIVERSE+SIGNALS:
        out[s] = get_index_daily_data(symbol=s, days=days)
    return out