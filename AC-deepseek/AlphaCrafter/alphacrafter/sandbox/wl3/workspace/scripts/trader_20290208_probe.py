"""Trader probe 2029-02-08: account state + regime snapshot at block start."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

acct = get_account_dict()
print("=== ACCOUNT ===")
print("net_assets:", acct.get("net_assets"))
print("total_assets:", acct.get("total_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} "
          f"cost={p['cost_price']:.4f} cur={p['current_price']:.4f} pnl={p['profit_loss']:.0f}")
print("pending orders:", len(acct.get("orders", [])))

def get_df(sym, days=260):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

print("\n=== REGIME SNAPSHOT (through last close) ===")
vix = get_df("VIX", 300)
if vix is not None:
    v = vix["close"].astype(float)
    print(f"VIX last={v.iloc[-1]:.1f} 5d_ago={v.iloc[-6]:.1f} 20d_ago={v.iloc[-21]:.1f} "
          f"60d_ago={v.iloc[-61]:.1f} 21d_chg={(v.iloc[-1]/v.iloc[-21]-1)*100:.1f}%")

watch = acct.get("watch_list", [])
rets = {}
closes = {}
for s in watch:
    df = get_df(s, 300)
    if df is None or len(df) < 30:
        print(f"{s}: NO DATA")
        continue
    c = df["close"].astype(float)
    closes[s] = c
    r = c.pct_change()
    rets[s] = r
    print(f"{s}: last={c.iloc[-1]:.2f} 5d={(c.iloc[-1]/c.iloc[-6]-1)*100:+.2f}% "
          f"21d={(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}% "
          f"60d={(c.iloc[-1]/c.iloc[-61]-1)*100:+.2f}% "
          f"rv20={r.tail(20).std()*100:.2f}%")

panel = pd.concat(rets, axis=1).dropna()
mkt = panel.mean(axis=1)
wealth = (1 + mkt).cumprod()
mdd = (wealth / wealth.rolling(60).max() - 1).tail(20).min()
print(f"\nmean 21d live ret: {panel.tail(21).mean().mean()*100:+.2f}%")
print(f"mkt20: {mkt.tail(20).mean()*100:+.3f}%  mdd20: {mdd*100:.2f}%")
print(f"vol20 mean: {panel.tail(20).std().mean()*100:.2f}%  vol_med120: {panel.tail(120).std().median()*100:.2f}%")

# frozen check
print("\n=== FROZEN CHECK (last 120d unique closes) ===")
for s in watch:
    if s in closes:
        q = closes[s].dropna().tail(120)
        if len(q) >= 20:
            print(f"{s}: unique={q.nunique()} last={q.iloc[-1]:.4f}")

# equity stress metrics
eq = [s for s in watch if s in ("000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX")]
if eq:
    eqr = np.mean([closes[s].iloc[-1]/closes[s].iloc[-22]-1 for s in eq if s in closes])
    print(f"\nlive-eq 21d mean ret: {eqr*100:+.2f}%")
    for s in eq:
        if s in closes:
            print(f"  {s}: 21d={(closes[s].iloc[-1]/closes[s].iloc[-22]-1)*100:+.2f}%")
