"""miner_1 exploration: new factor families + rank-composites for daily IC stability.
Universe: 15 cross-asset tradables. Gate: |IC1|>=0.0070 and |ICIR1|>=0.0840 (daily paper IC).
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from miner1_common import load_close, ic_analysis, coverage, turnover

t0 = time.time()
closes = load_close()
idx = None
for s in closes:
    idx = closes[s].index if idx is None else idx.intersection(closes[s].index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]  # 1y warmup
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in closes})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in closes})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in closes})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in closes})
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in closes})
RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))
EW = CP.mean(axis=1)
EWR = EW.pct_change()
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10)}
N_CELLS = len(idx) * len(closes)
SYMS = list(closes.keys())


def cov_to(factor_df):
    return coverage(factor_df, closes), turnover(factor_df)


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    c, t = cov_to(panel)
    ic1 = ic_analysis(panel, closes, fwd_days=1)
    ic5 = ic_analysis(panel, closes, fwd_days=5)
    ic10 = ic_analysis(panel, closes, fwd_days=10)
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:26s} cov={c:.3f} to={t:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": c, "to": t, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


panels = {}

# ---------- 1) Trend efficiency (Kaufman ER): |C-C[n]| / sum(|dC|) ----------
for n in (10, 20, 60, 120):
    panels[f"eff_ratio_{n}d"] = (CP - CP.shift(n)).abs() / LRET.abs().rolling(n).sum()

# ---------- 2) Trend slope t-stat (linear regression on last n days) ----------
def slope_tscore(n):
    def f(s):
        s = s.dropna()
        x = np.arange(n, dtype=float)
        xm = x - x.mean()
        xx = (xm ** 2).sum()
        out = {}
        for i in range(n - 1, len(s)):
            y = s.iloc[i - n + 1:i + 1].values.astype(float)
            ym = y - y.mean()
            b = (xm * ym).sum() / xx
            resid = y - (y.mean() + b * xm)
            se = np.sqrt((resid ** 2).sum() / (n - 2) / xx)
            out[s.index[i]] = b / se if se > 0 else np.nan
        return pd.Series(out)
    return f

for n in (30, 60, 120):
    cols = {}
    for s in SYMS:
        cols[s] = slope_tscore(n)(CP[s])
    panels[f"slope_tscore_{n}d"] = pd.DataFrame(cols)

# ---------- 3) Win-rate / consistency ----------
for n in (60, 120, 250):
    panels[f"pos_freq_{n}d"] = (LRET > 0).rolling(n).mean()

# ---------- 4) Volume factors ----------
lvol = np.log(VOL + 1e-9)
panels["vol_trend_20d"] = lvol.rolling(20).mean() - lvol.rolling(60).mean()   # volume regime shift
for n in (10, 20, 60):
    panels[f"vol_z_{n}d"] = (lvol - lvol.rolling(n).mean()) / lvol.rolling(n).std()  # attention spike
# Amihud illiquidity (negative: illiquid -> premium)
amihud = (LRET.abs() / (VOL + 1e-9))
for n in (20, 60):
    panels[f"amihud_neg_{n}d"] = -np.log(amihud.rolling(n).mean() + 1e-15)
# OBV slope normalized
obv = (np.sign(LRET) * VOL).cumsum()
for n in (20, 60):
    panels[f"obv_slope_{n}d"] = (obv.rolling(n).mean() - obv.shift(n).rolling(n).mean()) / (VOL.rolling(n).mean() + 1e-9)

# ---------- 5) Relative strength vs EW index ----------
for n in (20, 60, 120):
    panels[f"rs_{n}d"] = (LRET.rolling(n).sum() - EWR.rolling(n).sum())

# ---------- 6) Macro-conditional interactions ----------
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")["close"].reindex(idx)
spx = CP["SPX"]
spx_trend = (spx / spx.shift(20) - 1.0)  # SPX 20d trend
vix_trend = vix / vix.shift(20) - 1.0

def roll_beta(y, x, win):
    m = pd.concat([y, x], axis=1).dropna()
    yv, xv = m.iloc[:, 0], m.iloc[:, 1]
    return (yv.rolling(win).cov(xv) / xv.rolling(win).var()).reindex(idx)

b_spx = {w: pd.DataFrame({s: roll_beta(RET[s], spx.pct_change(), w) for s in SYMS}) for w in (60, 120)}
b_vix = {w: pd.DataFrame({s: roll_beta(RET[s], vix.pct_change(), w) for s in SYMS}) for w in (60, 120)}

panels["beta_spx60_x_spxtrend"] = b_spx[60] * np.sign(spx_trend)          # risk-on tilt
panels["beta_vix60_x_vixret"] = -b_vix[60] * np.sign(vix_trend)          # defensive tilt when VIX rising
panels["beta_spx60_x_vixlevel"] = b_spx[60] * np.sign(vix.rolling(20).mean() - vix.rolling(120).mean())

# ---------- 7) Weekday seasonality persistence ----------
wd = idx.dayofweek
wd_hist = LRET.groupby(wd).transform(lambda x: x.rolling(26, min_periods=13).mean())  # avg ret for that weekday
panels["weekday_hist_26w"] = wd_hist

# ---------- 8) Turn-of-month ----------
dom = idx.day
days_in_m = idx.days_in_month
panels["tom_proximity"] = -((days_in_m - dom) / days_in_m)  # closer to month end -> larger
panels["tom_binary"] = ((days_in_m - dom) <= 2).astype(float)

# ---------- 9) Composite rank-averaged factors ----------
def zrank(panel):
    return panel.rank(axis=1)

mom = {n: LRET.rolling(n).sum() for n in (20, 60, 120, 250)}
norm_ma20_60 = (CP.rolling(20).mean() - CP.rolling(60).mean()) / CP.rolling(60).std()
close_vs_ma120 = CP / CP.rolling(120).mean() - 1.0
dist_52w_low = CP / CP.rolling(252).min() - 1.0

panels["comp_mom_rank"] = (zrank(mom[20]) + zrank(mom[60]) + zrank(mom[120]) + zrank(mom[250])) / 4
panels["comp_trend_mix"] = (zrank(mom[60].shift(5)) + zrank(norm_ma20_60) + zrank(close_vs_ma120) + zrank(dist_52w_low)) / 4
panels["comp_mom_er"] = (zrank(mom[60]) + zrank(panels["eff_ratio_60d"]) + zrank(panels["pos_freq_60d"])) / 3
panels["comp_mom_rs"] = (zrank(mom[60]) + zrank(panels["rs_60d"]) + zrank(panels["eff_ratio_60d"])) / 3
panels["comp_vol_adj_mom"] = (zrank(mom[60]) + zrank(-RET.rolling(60).std()) + zrank(panels["rs_60d"])) / 3
panels["comp_mom_vol_rs"] = (zrank(mom[60]) + zrank(norm_ma20_60) + zrank(-RET.rolling(60).std()) + zrank(panels["rs_60d"])) / 4

# ---------- 10) Regime-conditional momentum ----------
asset_vol60 = RET.rolling(60).std()
lo_vol_regime = (asset_vol60 < asset_vol60.rolling(252).median()).astype(float)
panels["mom60_lowvol"] = mom[60].shift(5) * lo_vol_regime   # trend only in calm regime
panels["mom60_highvol"] = mom[60].shift(5) * (1 - lo_vol_regime)
ew_up = (EWR.rolling(20).sum() > 0).astype(float)
panels["mom60_ewup"] = mom[60].shift(5) * ew_up              # trend only in risk-on EW regime
panels["switch_mom_rev"] = mom[60].shift(5) * ew_up + (-LRET.rolling(5).sum()) * (1 - ew_up)

res = [run(n, p) for n, p in panels.items()]
print(f"\nfinished {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)} passed gate")
for r in res:
    if r["passed"]:
        print("PASS:", r["name"], r["ic1"])
