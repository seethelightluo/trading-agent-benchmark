"""miner_1 2034-09-28 probe: current tape snapshot (data through 2034-09-27 visible).
Reports recent returns, VIX regime, dispersion to inform factor design for this cycle.
"""
import pandas as pd
import numpy as np

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load_close():
    out = {}
    for s in ASSETS:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
        df = df[df['date'] <= pd.Timestamp('2034-09-27')].set_index('date').sort_index()
        out[s] = df['close'].astype(float)
    panel = pd.DataFrame(out).sort_index()
    return panel

def macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp('2034-09-27')].set_index('date').sort_index()
    return df['close'].astype(float)

panel = load_close()
ret = panel.pct_change()
last = panel.index[-1]
print("panel dates:", panel.index[0].date(), "->", last.date(), "rows:", len(panel))

def r(days):
    return (panel.iloc[-1] / panel.iloc[-1 - days] - 1.0) * 100

for days in [5, 10, 20, 60, 180, 252]:
    rr = r(days).sort_values(ascending=False)
    print(f"\n=== {days}d return (%) ===")
    print(rr.round(2).to_string())

vix = macro('VIX')
print("\nVIX last 5:", vix.tail(5).round(1).to_dict())
print("VIX 20d mean:", round(vix.tail(20).mean(), 1), "60d max:", round(vix.tail(60).max(), 1))
print("VIX 60d ago:", round(vix.iloc[-61], 1))

dxy = macro('DXY')
print("DXY 20d chg %:", round((dxy.iloc[-1]/dxy.iloc[-21]-1)*100, 2))
usdjpy = macro('USDJPY')
print("USDJPY 20d chg %:", round((usdjpy.iloc[-1]/usdjpy.iloc[-21]-1)*100, 2))

# dispersion
r20 = r(20)
print("\n20d dispersion (max-min pp):", round(r20.max()-r20.min(), 1))
print("252d range position:")
hp = panel.rolling(252, min_periods=60).max()
print(((panel.iloc[-1] - panel.rolling(60).min().iloc[-1]) / (hp.iloc[-1] - panel.rolling(60).min().iloc[-1])).round(3).to_string())
