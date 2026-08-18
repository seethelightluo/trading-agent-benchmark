"""Trader probe: reconstruct 20281214-20281228 block details for memory logging."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WL = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def getter(sym, days=60):
    for fn in (get_stock_daily_data, get_index_daily_data):
        try:
            df = fn(symbol=sym, days=days)
            if df is not None and len(df):
                return df
        except Exception:
            pass
    return None


# --- Block return per asset: 2028-12-14 close -> 2028-12-28 close (visible 12-27) ---
print("=== per-asset block returns (close 12-13 -> close 12-27) ===")
closes = {}
for s in WL:
    df = getter(s, 30)
    if df is None or len(df) < 15:
        print(s, "NO DATA")
        continue
    df = df.sort_values('date')
    closes[s] = df['close'].astype(float)
    c_last = float(df['close'].iloc[-1])
    c_14 = float(df[df['date'] <= '2028-12-14']['close'].iloc[-1]) if (df['date'] <= '2028-12-14').any() else None
    if c_14:
        print(f"{s}: close_14={c_14:.4f} close_last={c_last:.4f} ret={(c_last/c_14-1)*100:+.2f}%")

# --- Regime at decision date 2028-12-14 (trend using data visible then) ---
print("\n=== regime probe at 2028-12-14 decision ===")
frames = {s: getter(s, 300) for s in WL}
panel = None
for s, df in frames.items():
    if df is None or len(df) < 140:
        continue
    df = df.sort_values('date')
    df = df[df['date'] <= '2028-12-14']
    c = df['close'].astype(float).rename(s)
    panel = c if panel is None else pd.concat([panel, c], axis=1, join='inner')
if panel is not None and len(panel) > 30:
    rets = panel.pct_change().dropna()
    mkt = rets.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
    regime = 'bull' if trend > 1.0 else ('bear' if trend < -1.0 else 'sideways')
    print(f"trend={trend:.3f} regime={regime}")
else:
    print("insufficient panel")

# --- Trend at block end (12-27 visible) ---
print("\n=== regime probe at 2028-12-27 (block end) ===")
panel2 = None
for s, df in frames.items():
    if df is None or len(df) < 140:
        continue
    df = df.sort_values('date')
    c = df['close'].astype(float).rename(s)
    panel2 = c if panel2 is None else pd.concat([panel2, c], axis=1, join='inner')
if panel2 is not None and len(panel2) > 30:
    rets = panel2.pct_change().dropna()
    mkt = rets.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
    regime = 'bull' if trend > 1.0 else ('bear' if trend < -1.0 else 'sideways')
    print(f"trend={trend:.3f} regime={regime}")

# --- NAV path check ---
acct = json.load(open('../persistent/account.json'))
print("\nnet_assets now:", round(acct['net_assets'], 2))
print("last_rebalance_date:", acct.get('last_rebalance_date'))
print("cumulative_transaction_cost:", round(acct.get('cumulative_transaction_cost', 0), 2))
