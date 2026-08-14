"""Trader block analysis: 2035-01-04 -> 2035-01-18 live block attribution.

Uses the post-step account state (current) and daily data to reconstruct the
block-start (01-04 rebalance) weights and attribute the period return.
"""
import json

import pandas as pd

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}


def get_df(symbol, days=60):
    try:
        if symbol in OBS:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None


def main():
    acct = json.load(open("../persistent/account.json"))
    nav = acct["net_assets"]
    cur = {p["symbol"]: p["market_value"] for p in acct["positions"]}

    # current date per sim data: find last close in each frame
    r = {}
    for a in ASSETS:
        df = get_df(a, days=45)
        if df is None or len(df) < 30:
            print(a, "NO DATA")
            continue
        s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
        r[a] = s

    # block returns: 01-04 close -> 01-18 close (last two closes)
    rets = {}
    for a, s in r.items():
        if len(s) >= 2:
            rets[a] = float(s.iloc[-1] / s.iloc[-11] - 1.0)  # 10 trading days
        else:
            rets[a] = 0.0

    # reconstruct block-start weights: w0 = cur_mv/(1+r) normalized
    mv0 = {a: cur.get(a, 0.0) / (1.0 + rets.get(a, 0.0)) for a in ASSETS}
    tot0 = sum(mv0.values())
    w0 = {a: mv0[a] / tot0 for a in ASSETS}

    contrib = {a: w0[a] * rets[a] for a in ASSETS}
    total = sum(contrib.values())

    print(f"net_assets now: {nav:,.2f}")
    print(f"approx block-start NAV: {tot0:,.2f}")
    print(f"\n{'asset':10s} {'w0%':>7s} {'ret%':>8s} {'contrib%':>9s}")
    print("-" * 40)
    for a in ASSETS:
        print(f"{a:10s} {w0[a]*100:7.2f} {rets[a]*100:8.2f} {contrib[a]*100:9.3f}")
    print("-" * 40)
    print(f"{'TOTAL':10s} {'100.00':>7s} {'':>8s} {total*100:9.3f}")

    # order book / pending
    print("\nn_positions:", len(acct["positions"]))
    print("orders:", len(acct.get("orders", [])))
    print("cash:", acct.get("available_cash"))

    # frozen pin check
    fr = ["HSI", "SX5E", "BTC", "US10Y", "CN10Y"]
    print("\nfrozen5 weights:", [f"{cur[a]/nav*100:.3f}%" for a in fr])


if __name__ == "__main__":
    main()
