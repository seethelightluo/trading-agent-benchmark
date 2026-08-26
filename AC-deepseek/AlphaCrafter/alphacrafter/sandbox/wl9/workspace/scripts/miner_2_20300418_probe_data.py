"""miner_2 2030-04-18 probe: data availability through visible date 2030-04-17."""
import pandas as pd
from pathlib import Path

TRADABLE = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
SD = Path('../persistent/stock_data')
print("=== Tradable assets ===")
for a in TRADABLE:
    p = SD / f'{a}.csv'
    if not p.exists():
        print(f'{a:10s} NO FILE'); continue
    df = pd.read_csv(p, parse_dates=['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    vol = pd.to_numeric(df['volume'], errors='coerce') if 'volume' in df.columns else None
    d = pd.to_datetime(df['date'])
    end = d.max()
    vnan = vol.isna().sum() if vol is not None else 'NA'
    vzero = (vol == 0).sum() if vol is not None else 'NA'
    print(f"{a:10s} len={len(df):5d} {d.min().date()} -> {end.date()} volNaN={vnan} volZero={vzero}")

print("\n=== Macro (index_data) ===")
ID = Path('../persistent/index_data')
for a in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    p = ID / f'{a}.csv'
    if not p.exists():
        print(f'{a:8s} NO FILE'); continue
    df = pd.read_csv(p, parse_dates=['date'])
    d = pd.to_datetime(df['date'])
    print(f"{a:8s} len={len(df):5d} {d.min().date()} -> {d.max().date()}")