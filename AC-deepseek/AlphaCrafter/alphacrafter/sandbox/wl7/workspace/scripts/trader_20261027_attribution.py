import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def series(sym):
    try:
        df = get_stock_daily_data(sym, days=40)
    except Exception:
        try:
            df = get_index_daily_data(sym, days=40)
        except Exception:
            return None
    if df is None or len(df) < 2:
        return None
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    return s

# block: 2026-10-13 close (decision) -> last available (10-27)
print(f"{'asset':<12} {'close@10-12':>12} {'close@10-27':>12} {'block_ret%':>10}")
rows = []
for a in assets:
    s = series(a)
    if s is None:
        print(a, "no data")
        continue
    last = s.iloc[-1]
    prev = s[s.index <= "2026-10-13"]
    if len(prev) == 0:
        print(a, "no pre-block data")
        continue
    base = prev.iloc[-1]
    r = (last / base - 1) * 100
    rows.append((a, base, last, r))
    print(f"{a:<12} {base:>12.1f} {last:>12.1f} {r:>10.2f}")

# VIX regime check at block end
try:
    v = series("VIX")
    if v is not None:
        print("\nVIX last 5:", v.tail(5).round(2).to_dict())
        v10 = series("VIX")
        if v10 is not None:
            prev = v10[v10.index <= "2026-10-13"]
            print("VIX at 10-13:", round(float(prev.iloc[-1]),2), "-> end:", round(float(v10.iloc[-1]),2))
except Exception as e:
    print("VIX err", e)

# SPX trend (mkt regime)
try:
    spx = series("SPX")
    if spx is not None:
        spx20 = spx[spx.index <= "2026-10-13"].tail(20)
        print("\nSPX 20d mean daily ret @10-13:", round(float(spx20.pct_change().mean())*100, 4), "%")
except Exception as e:
    print("SPX err", e)
