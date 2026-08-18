"""Trader regime check at 2027-04-13 decision (data thru 2027-04-12)."""
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def series(sym, days=300):
    try:
        df = get_stock_daily_data(sym, days=days)
    except Exception:
        df = None
    if df is None or len(df) < 30:
        try:
            df = get_index_daily_data(sym, days=days)
        except Exception:
            df = None
    if df is None or len(df) < 30:
        return None
    s = pd.Series(df['close'].astype(float), index=pd.to_datetime(df['date']))
    return s.sort_index()

print("=== watchlist 20d/60d momentum & c/MA20 (thru %s) ===" % series('SPX').index[-1].date())
for a in WATCH:
    s = series(a)
    if s is None:
        print(f"{a:10s} NO DATA")
        continue
    m20 = s.iloc[-1]/s.iloc[-21]-1 if len(s) > 21 else float('nan')
    m60 = s.iloc[-1]/s.iloc[-61]-1 if len(s) > 61 else float('nan')
    ma20 = s.rolling(20).mean().iloc[-1]
    cm = s.iloc[-1]/ma20 if ma20 and ma20 > 0 else float('nan')
    last = s.iloc[-1]
    print(f"{a:10s} last={last:12.4f} m20={m20*100:8.2f}% m60={m60*100:8.2f}% c/MA20={cm:6.3f}")

print("\n=== macro observables ===")
for a in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = series(a)
    if s is None:
        print(f"{a:8s} NO DATA")
        continue
    m5 = s.iloc[-1]/s.iloc[-6]-1 if len(s) > 6 else float('nan')
    m20 = s.iloc[-1]/s.iloc[-21]-1 if len(s) > 21 else float('nan')
    print(f"{a:8s} last={s.iloc[-1]:10.4f} m5={m5*100:7.2f}% m20={m20*100:7.2f}%  last5={[round(x,2) for x in s.tail(5)]}")

# equity mkt trend
rets = {}
for a in WATCH:
    s = series(a)
    if s is not None:
        rets[a] = s.pct_change()
R = pd.concat(rets, axis=1, join='inner').dropna().tail(150)
mkt = R.mean(axis=1)
print("\n=== regime ===")
print("mkt_20d mean daily:", round(float(mkt.tail(20).mean())*100, 4), "%")
print("mkt_60d mean daily:", round(float(mkt.tail(60).mean())*100, 4), "%")
print("mean 20d |corr|:", round(float(R.tail(20).corr().abs().values.mean()), 4))
print("20d cross-sectional disp (std of daily cross-section, mean):", round(float(R.tail(20).std(axis=1).mean())*100, 3), "%")
print("60d cross-sectional disp:", round(float(R.tail(60).std(axis=1).mean())*100, 3), "%")
print("20d vol (EW mkt):", round(float(mkt.tail(20).std())*100, 3), "%")
