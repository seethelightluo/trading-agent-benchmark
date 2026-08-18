"""Screener analysis 2029-06-19 cycle: compute factor exposures as of visible 2029-06-18."""
import pandas as pd, numpy as np, json

VISIBLE = "2029-06-18"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = ["DXY","EURUSD","VIX"]

def load(sym, is_index=False):
    p = f"../persistent/{'index_data' if is_index else 'stock_data'}/{sym}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date").reset_index(drop=True)
    return df

# Build aligned return panel
R = None
for a in ASSETS:
    df = load(a)
    s = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]))
    r = s.pct_change().rename(a)
    R = r if R is None else pd.concat([R, r], axis=1)
R = R.dropna().tail(200)
cp = (1.0 + R).cumprod()
mkt = R.mean(axis=1)

def ranks(series):
    valid = series.dropna()
    out = pd.Series(0.5, index=series.index)
    if len(valid) == 0: return out
    rk = valid.rank(pct=True)
    out.loc[rk.index] = rk
    return out

# Factors (mirror strategy.py)
mom = cp.shift(5) / cp.shift(25) - 1.0
rel_mom = mom.sub(mom.median(axis=1), axis=0)

mvar = mkt.rolling(60).var()
beta_ew = R.rolling(60).cov(mkt).div(mvar, axis=0)

neg = R.clip(upper=0.0)
semi = (neg ** 2).rolling(20).mean() ** 0.5
tot = R.rolling(20).std()
dvr = -(semi / tot)

mx = R.rolling(20).max()

corr_parts = []
for a in R.columns:
    others = [R[a].rolling(60).corr(R[b]) for b in R.columns if b != a]
    corr_parts.append(pd.concat(others, axis=1).mean(axis=1).rename(a))
corr_ew = pd.concat(corr_parts, axis=1)

kurt = R.shift(5).rolling(20).kurt()

dxy = load("DXY", True)
dc = pd.Series(dxy["close"].astype(float).values, index=pd.to_datetime(dxy["date"]))
dxy_ret = dc.pct_change().reindex(R.index)
dxy_20 = (dc / dc.shift(20) - 1.0).reindex(R.index)
dvar = dxy_ret.rolling(60).var()
bfx = R.rolling(60).cov(dxy_ret).div(dvar, axis=0)
dxy_cond = -bfx * dxy_20

eur = load("EURUSD", True)
ec = pd.Series(eur["close"].astype(float).values, index=pd.to_datetime(eur["date"]))
eur_ret = ec.pct_change().reindex(R.index)
eur_20 = (ec / ec.shift(20) - 1.0).reindex(R.index)
evar = eur_ret.rolling(60).var()
bfx_e = R.rolling(60).cov(eur_ret).div(evar, axis=0)
eur_cond = bfx_e * eur_20

vix = load("VIX", True)
vix_last = vix["close"].iloc[-1]

factors = {
    "rel_mom_20d_skip5": rel_mom,
    "beta_ew_60d": beta_ew,
    "downside_vol_ratio_20": dvr,
    "max_ret_20d": mx,
    "dxy_beta_cond_60x20": dxy_cond,
    "corr_ew_60": corr_ew,
    "kurt_20d_skip5": kurt,
    "eurusd_beta_cond_60x20": eur_cond,
}

print("VIX last:", round(vix_last,2))
print("DXY 20d:", round(dxy_20.iloc[-1]*100,2), "%  EURUSD 20d:", round(eur_20.iloc[-1]*100,2), "%")
print("mkt_20:", round(mkt.tail(20).mean()*10000,2), "bp/day")
print()

# 30d/60d returns for context
print("=== 30d & 60d returns ===")
for a in ASSETS:
    s = R[a]
    r30 = (1+s.tail(30)).prod()-1
    r60 = (1+s.tail(60)).prod()-1
    flat = "FROZEN" if abs(r30) < 1e-9 and abs(r60) < 1e-9 else ""
    print(f"{a:12s} 30d {r30*100:8.2f}%  60d {r60*100:8.2f}%  {flat}")

print()
print("=== Factor cross-sectional ranks (0..1) as of last visible date ===")
out = {}
for fid, fr in factors.items():
    last = fr.iloc[-1]
    rk = ranks(last)
    out[fid] = rk
    top = rk.sort_values(ascending=False)
    bot = rk.sort_values()
    print(f"\n{fid}")
    print("  top3:", [(a, round(v,2)) for a,v in top.head(3).items()])
    print("  bot3:", [(a, round(v,2)) for a,v in bot.head(3).items()])

# Correlation of factor rank vectors (crowding check)
print("\n=== Factor rank correlation matrix ===")
fk = pd.DataFrame({k: v for k, v in out.items()})
print(fk.corr().round(2).to_string())
