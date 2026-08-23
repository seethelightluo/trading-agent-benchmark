"""Probe data + fundamentals columns coverage through current sim date (2028-10-05).
Visible data = through previous completed trading day.
Focus: PE, PS, PB, DYR presence in stock daily data; macro index data availability.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== Tradable assets (stock API), cols detail ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is not None and len(df):
        cols = list(df.columns)
        has_fund = [c for c in ['PE','PS','PB','DYR'] if c in cols]
        fund_note = f"fundcols={has_fund}"
        # coverage of fundamental cols
        cov = {}
        for c in has_fund:
            nonnull = df[c].notna().sum()
            cov[c] = f"{nonnull}/{len(df)} ({nonnull/len(df):.2%})"
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} | {fund_note} | {cov}")
    else:
        print(f"{sym:10s} NO DATA")

print("\n=== Macro (index API) ===")
for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_index_daily_data(symbol=sym, days=4000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} cols={list(df.columns)}")
    else:
        print(f"{sym}: NONE")

print("\n=== Latest date per asset ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=10)
    if df is not None and len(df):
        print(f"{sym:10s} last={df['date'].iloc[-1].date()}")