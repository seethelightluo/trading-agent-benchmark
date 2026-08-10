import pandas as pd, numpy as np, os
DATA="../persistent/stock_data"; IDX="../persistent/index_data"
SYMS=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END="2026-07-15"

def load(s, d=DATA):
    df=pd.read_csv(os.path.join(d,f"{s}.csv"))
    df["date"]=pd.to_datetime(df["date"])
    return df[df["date"]<=END].set_index("date").sort_index()

closes={s:load(s)["close"] for s in SYMS}
macro={s:load(s,IDX)["close"] for s in MACRO}

print("=== last date available:", max(closes["SPX"].index))
print("\n=== 21d / 63d / 126d returns (to 2026-07-15) ===")
for s in SYMS:
    c=closes[s]
    r21=c.iloc[-1]/c.iloc[-22]-1; r63=c.iloc[-1]/c.iloc[-64]-1; r126=c.iloc[-1]/c.iloc[-127]-1
    print(f"{s:10s} r21={r21:+.3f} r63={r63:+.3f} r126={r126:+.3f}")

print("\n=== trend: price vs MA50/MA200, MA slope ===")
for s in SYMS:
    c=closes[s]
    ma50=c.rolling(50).mean().iloc[-1]; ma200=c.rolling(200).mean().iloc[-1]
    slope50=(c.rolling(50).mean().iloc[-1]/c.rolling(50).mean().iloc[-21]-1)
    print(f"{s:10s} px={c.iloc[-1]:>12.2f} ma50={ma50:>12.2f} ma200={ma200:>12.2f} px>ma50={c.iloc[-1]>ma50} px>ma200={c.iloc[-1]>ma200} slope50={slope50:+.4f}")

print("\n=== realized vol (ann., 21d & 63d) & 252d max DD ===")
for s in SYMS:
    c=closes[s]; r=c.pct_change()
    v21=r.tail(21).std()*np.sqrt(252); v63=r.tail(63).std()*np.sqrt(252)
    dd=(c/c.cummax()-1).tail(252).min()
    print(f"{s:10s} vol21={v21:.2%} vol63={v63:.2%} maxDD252={dd:.2%}")

print("\n=== macro obs (last obs <= 2026-07-15) ===")
for s in MACRO:
    m=macro[s]
    chg21=m.iloc[-1]/m.iloc[-22]-1
    print(f"{s:8s} last={m.iloc[-1]:>10.3f} chg21={chg21:+.3f} 60d_mean={m.tail(60).mean():.3f}")

ret=pd.DataFrame({s:closes[s].pct_change() for s in SYMS}).dropna()
c63=ret.tail(63).corr()
avg_corr=(c63.values.sum()-len(c63))/(len(c63)*(len(c63)-1))
print(f"\n=== avg pairwise corr (63d, 15 assets) = {avg_corr:.3f}")
print("=== avg pairwise corr (21d) ===", f"{(ret.tail(21).corr().values.sum()-15)/(15*14):.3f}")
