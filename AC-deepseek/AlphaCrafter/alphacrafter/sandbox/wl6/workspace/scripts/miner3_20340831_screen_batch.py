"""miner_3 screen batch (2034-08-31). Fresh candidate factors for the 15-instrument
cross-asset universe, as-of visible_through 2034-08-30. Uses CSV files directly
(same data the simulator reads) to include macro observation series.

Admission gate (benchmark, shared): |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Also report recent-window (last 2y) instability for citizen over-rotation.
"""
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro
import pandas as pd, numpy as np, glob

VIS = "2034-08-30"
TRADABLE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
            "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

# ---- load tradable panel ----
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill().dropna()
ret = px.pct_change()

def evalc(fname, f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return
    icm = ic.mean(); icstd = ic.std(ddof=1)
    icir = icm/icstd if icstd and icstd>0 else np.nan
    hit = (ic>0).mean()
    recent = ic[ic.index >= "2032-09-01"]
    ricm = recent.mean() if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = f.notna().mean().mean()
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent_2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

# A. risk-scaled momentum 20d
mom20 = px/px.shift(20) - 1.0
rv20 = ret.rolling(20).std()
cands["vol_scaled_mom_20"] = (mom20/rv20.replace(0,np.nan))

# B. price distance from sma20 risk-normalized
sma20 = px.rolling(20).mean()
cands["ma_dist_rz_20"] = ((px-sma20)/(rv20*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)

# C. breakout 40 proximity
cands["breakout_40"] = px/px.rolling(40).max() - 1.0

# D. breadth up days 20d
up = (ret>0).rolling(20).mean()
cands["breadth_20"] = up - 0.5

# E. cross-asset: VIX change vs asset correlation regime (mean-reversion): negative asset beta to VIX chg
vix = load_macro("VIX", VIS)
dvix = vix.diff().reindex(px.index).ffill()
vix_mom = (vix/vix.shift(10)-1).reindex(px.index).ffill()
# asset sensitivity to VIX over 60d -> then negate (returns low when VIX rises)
dv_all = pd.concat([dvix], axis=1); dv_all.columns=["VIX"]
beta_vix = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    bx = px[a].rolling(60).corr(vix_mom)
    beta_vix[a] = bx
cands["vix_mom_beta_60"] = -beta_vix

# F. DXY momentum cross-sectional tilt: assets that rose when DXY fell (carry)
dxy = load_macro("DXY", VIS)
dmom = (dxy/dxy.shift(20)-1).reindex(px.index).ffill()
dxy_beta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    dxy_beta[a] = px[a].pct_change().rolling(60).corr(dmom)
cands["dxy_beta_60_neg"] = -dxy_beta

for name, f in cands.items():
    evalc("", f, name)