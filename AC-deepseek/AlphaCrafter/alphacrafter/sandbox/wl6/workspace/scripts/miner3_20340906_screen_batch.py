"""miner_3 screen batch (2034-09-06), VIS 2034-09-05. Fresh candidate factors.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-09-05"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill()
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1])

def evalc(f, label):
    assert isinstance(f, pd.DataFrame), f"{label} is not a DataFrame"
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES")
        return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-09-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

# A. Cross-asset momentum 40d skip10
cands["mom_40_skip10"] = px.shift(10)/px.shift(50) - 1.0

# B. Trend: 90d change normalized by 20d vol
mom90 = px/px.shift(90) - 1.0
rv20 = ret.rolling(20).std()
cands["mom90_volnorm_20"] = mom90 / rv20.replace(0, np.nan)

# C. USDCNY beta (CNY-fixed risk): assets that fall when USDCNY rises
cny = load_macro("USDCNY", VIS).reindex(px.index).ffill()
cnym = (cny/cny.shift(20)-1).reindex(px.index).ffill()
cnybeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    cnybeta[a] = ret[a].rolling(60).corr(cnym)
cands["cny_beta_60_neg"] = -cnybeta

# D. Cross-sectional vol percentile vs own 1y history
rv20b = ret.rolling(20).std()
cands["vol_pctile_20x250"] = rv20b.rolling(250).rank(pct=True)

# E. 5d reversal
cands["reversal_5"] = -(px/px.shift(5)-1)

# F. EURUSD beta weak-dollar carry (assets rising when EUR rises)
eur = load_macro("EURUSD", VIS).reindex(px.index).ffill()
emom = (eur/eur.shift(20)-1).reindex(px.index).ffill()
ebeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    ebeta[a] = ret[a].rolling(60).corr(emom)
cands["eur_beta_60"] = ebeta

for name, f in cands.items():
    evalc(f, name)