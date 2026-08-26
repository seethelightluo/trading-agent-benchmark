"""miner_1 cycle 2031-12-11: probe fundamental column fill rates (no lookahead)."""
import pandas as pd
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = pd.Timestamp('2031-12-11')

print("--- fundamental column fill rates (through 2031-12-11) ---")
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= CUTOFF]
    n = len(df)
    parts = []
    for c in ['PE','PS','PB','DYR']:
        if c in df.columns:
            nn = df[c].notna().sum()
            parts.append(f"{c}={nn}({nn/n*100:.0f}%)")
        else:
            parts.append(f"{c}=NA")
    print(f"{a:9s} n={n:5d} " + " ".join(parts))

print("\n--- sample of PE/PB/DYR for a few assets (last 5 rows) ---")
for a in ['000300.SH','SPX','BTC','XAU','US10Y']:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= CUTOFF]
    print(f"--- {a} ---")
    print(df[['date','close','PE','PS','PB','DYR']].tail(5).to_string())