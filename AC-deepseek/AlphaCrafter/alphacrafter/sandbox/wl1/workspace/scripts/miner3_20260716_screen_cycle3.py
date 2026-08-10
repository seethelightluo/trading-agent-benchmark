"""miner_3 cycle-3 broad screen (2026-07-16).

Fresh candidate families, avoiding previously screened reversal/CLV/correlation
families. Admission gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 (1d forward,
rank IC, 2021-01-04..2026-07-15, min 8 names/date).
"""
import time, sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, IDX_DIR

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
idx = idx[(idx <= pd.Timestamp("2026-07-15"))]

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})

RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))
vol5 = LRET.rolling(5).std() * np.sqrt(252)
vol20 = LRET.rolling(20).std() * np.sqrt(252)
vol60 = LRET.rolling(60).std() * np.sqrt(252)
mom20 = CP / CP.shift(20) - 1.0
mom60 = CP / CP.shift(60) - 1.0
N_CELLS = len(idx) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840
EVAL_START = pd.Timestamp("2021-01-04")

HORIZONS = (1, 2, 3, 5, 10, 20)
ret_shift = RET.shift(-1)
fwd = {h: RET.shift(-h) for h in HORIZONS}
fwd_ranks = {h: fwd[h].rank(axis=1) for h in HORIZONS}


def row_spearman(F, R):
    """Row-wise Spearman (rank-rank Pearson) with min-valid mask. Vectorized."""
    X = F.values.astype(float)
    Y = R.values.astype(float)
    valid = (~np.isnan(X)) & (~np.isnan(Y))
    n = valid.sum(axis=1)
    X = np.where(valid, X, np.nan)
    Y = np.where(valid, Y, np.nan)
    with np.errstate(all="ignore"):
        xm = np.nanmean(X, axis=1, keepdims=True)
        ym = np.nanmean(Y, axis=1, keepdims=True)
        xc = np.where(valid, X - xm, 0.0)
        yc = np.where(valid, Y - ym, 0.0)
        num = (xc * yc).sum(axis=1)
        dx = np.sqrt((xc * xc).sum(axis=1))
        dy = np.sqrt((yc * yc).sum(axis=1))
        corr = num / (dx * dy)
    corr = np.where((n >= 8) & np.isfinite(corr), corr, np.nan)
    return pd.Series(corr, index=F.index)


def evaluate(name, fac):
    out = {}
    fr = fac.rank(axis=1)
    for h in HORIZONS:
        s = row_spearman(fr, fwd_ranks[h])
        s = s[(s.index >= EVAL_START)].dropna()
        if len(s) < 120:
            out[h] = None
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=1))
        out[h] = dict(ic=m, icir=m / sd if sd > 1e-12 else 0.0,
                      hit=float((s > 0).mean()), n=int(len(s)))
    sub = fac.loc[fac.index >= EVAL_START]
    cov = float(sub.notna().mean().mean()) if len(sub) else 0.0
    rk = fac.rank(axis=1, pct=True)
    turn = float((rk - rk.shift(10)).abs().mean().mean()) if len(rk) else np.nan
    return dict(horizons=out, coverage=cov, turnover_10d=turn)


F = {}

# ---------- A. Trend quality / efficiency (momentum, not reversal) ----------
F["eff_ratio_20"] = mom20.abs() / LRET.abs().rolling(20).sum()
F["eff_ratio_60"] = mom60.abs() / LRET.abs().rolling(60).sum()
F["rlr_20"] = (LRET.clip(lower=0).rolling(20).sum() + LRET.clip(upper=0).rolling(20).sum().abs()) \
              / (LRET.abs().rolling(20).sum() + 1e-12)   # (gain+loss_abs)/total -> 1 when balanced, 2 all-up
F["rlr_20_cent"] = F["rlr_20"] - 1.0
F["mom20_sharpe"] = mom20 / (vol20 + 1e-12)
F["mom60_sharpe"] = mom60 / (vol60 + 1e-12)
F["mom20_60_spread"] = mom20 - mom60

# ---------- B. Return path structure ----------
def roll_autocorr(x, lag=1, win=20):
    a = x.rolling(win).corr(x.shift(lag))
    return a
F["autocorr_1_20"] = roll_autocorr(LRET, 1, 20)
F["autocorr_1_60"] = roll_autocorr(LRET, 1, 60)
F["updown_ratio_10"] = (LRET > 0).rolling(10).sum() / 10.0 - 0.5
F["updown_ratio_20"] = (LRET > 0).rolling(20).sum() / 20.0 - 0.5
F["gap_1d"] = OP / CP.shift(1) - 1.0                      # overnight gap
F["intraday_1d"] = CP / OP - 1.0                          # intraday move
F["gap_intra_agree"] = (OP / CP.shift(1) - 1.0) * (CP / OP - 1.0)  # follow-through

# ---------- C. Volatility structure ----------
F["vol_term_20_60"] = vol20 / (vol60 + 1e-12) - 1.0
F["vol_term_5_20"] = vol5 / (vol20 + 1e-12) - 1.0
F["inv_vol20"] = -1.0 / (vol20 + 1e-9)                    # low-vol long
F["range_squeeze_5_60"] = (HP - LP).rolling(5).mean() / ((HP - LP).rolling(60).mean() + 1e-12)
F["vol20_rank_120"] = vol20.rolling(120).rank(pct=True)   # low percentile -> long
F["vol20_rank_120_rev"] = -F["vol20_rank_120"]

# ---------- D. Volume / liquidity structure ----------
F["vol_price_corr_20"] = RET.rolling(20).corr(VO.pct_change())
F["obv_slope_20"] = (np.sign(LRET) * VO).rolling(20).mean() / (VO.rolling(20).mean() + 1e-9)
F["dollar_vol_trend_20"] = (CP * VO).pct_change(20)
F["volume_z_20"] = (VO - VO.rolling(20).mean()) / (VO.rolling(20).std() + 1e-9)

# ---------- E. Macro-conditioned (per-asset sensitivity x macro state) ----------
def load_macro(name):
    d = pd.read_csv(os.path.join(IDX_DIR, f"{name}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    s = d["close"].reindex(idx).astype(float)
    return s

dxy = load_macro("DXY"); vix = load_macro("VIX")
usdcny = load_macro("USDCNY"); usdjpy = load_macro("USDJPY")
eurusd = load_macro("EURUSD")
dxy_r = dxy.pct_change(); vix_r = vix.pct_change()
usdcny_r = usdcny.pct_change(); usdjpy_r = usdjpy.pct_change()
dxy_mom20 = dxy.pct_change(20); vix_chg20 = vix.pct_change(20)
usdcny_mom20 = usdcny.pct_change(20); usdjpy_mom20 = usdjpy.pct_change(20)


def roll_beta(y, x, win):
    """per-asset rolling beta of y (panel) on x (series)."""
    out = {}
    for s in y.columns:
        df = pd.concat([y[s], x], axis=1).dropna()
        cov = df.iloc[:, 0].rolling(win).cov(df.iloc[:, 1])
        var = df.iloc[:, 1].rolling(win).var()
        out[s] = (cov / (var + 1e-12)).reindex(idx)
    return pd.DataFrame(out)


F["dxy_beta_60"] = roll_beta(RET, dxy_r, 60)
F["dxy_beta_60_x_dxy_mom20"] = F["dxy_beta_60"].mul(dxy_mom20.to_numpy().reshape(-1, 1), axis=0)
F["vix_beta_60"] = roll_beta(RET, vix_r, 60)
F["vix_beta_60_x_vix_chg20"] = F["vix_beta_60"].mul(vix_chg20.to_numpy().reshape(-1, 1), axis=0)
F["usdcny_beta_60"] = roll_beta(RET, usdcny_r, 60)
F["usdcny_beta_60_x_mom20"] = F["usdcny_beta_60"].mul(usdcny_mom20.to_numpy().reshape(-1, 1), axis=0)
F["usdjpy_beta_60_x_mom20"] = roll_beta(RET, usdjpy_r, 60).mul(usdjpy_mom20.to_numpy().reshape(-1, 1), axis=0)
F["vix_level_cond"] = pd.DataFrame(-vix.reindex(idx).to_numpy().reshape(-1, 1) * (vol60.to_numpy() + 1e-9), index=idx, columns=SYMBOLS)  # low VIX & low vol -> long

# ---------- F. Market-relative (equal-weight cross-asset index) ----------
mkt = RET.mean(axis=1)
mkt_mom20 = mkt.pct_change(20)
F["beta_mkt_60"] = roll_beta(RET, mkt, 60)
F["beta_mkt_60_x_mkt_mom20"] = F["beta_mkt_60"].mul(mkt_mom20.to_numpy().reshape(-1, 1), axis=0)

# ---------- G. Calendar / weekday persistence ----------
dow = CP.index.dayofweek.to_numpy()
dow_avg = {}
for s in SYMBOLS:
    sr = LRET[s]
    tmp = pd.DataFrame({"r": sr, "dow": dow}).dropna()
    g = tmp.groupby("dow")["r"].mean()
    dow_avg[s] = g.reindex(range(5)).to_numpy()
dow_avg = pd.DataFrame(dow_avg, index=pd.RangeIndex(5))  # rows: Mon..Fri
dow_avg_map = dow_avg.to_numpy().T
F["dow_avg_ret"] = pd.DataFrame(dow_avg_map.T[dow], index=idx, columns=SYMBOLS)

results = {name: evaluate(name, fac) for name, fac in F.items()}

print(f"{'factor':26s} {'IC1':>8s} {'ICIR1':>8s} {'hit1':>6s} {'n1':>5s} {'IC5':>8s} {'IC10':>8s} {'IC20':>8s} {'cov':>6s} {'turn10':>7s}  gate")
rows = []
for name, r in results.items():
    h = r["horizons"]
    g = h.get(1)
    if g is None:
        continue
    h5, h10, h20 = h.get(5), h.get(10), h.get(20)
    passed = abs(g["ic"]) >= GATE_IC and abs(g["icir"]) >= GATE_ICIR
    rows.append((name, g["ic"], g["icir"], g["hit"], g["n"],
                 h5["ic"] if h5 else np.nan, h10["ic"] if h10 else np.nan,
                 h20["ic"] if h20 else np.nan, r["coverage"], r["turnover_10d"], passed))
rows.sort(key=lambda x: -abs(x[1]))
for r in rows:
    mark = "PASS" if r[-1] else "   "
    print(f"{r[0]:26s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.3f} {r[4]:5d} {r[5]:8.4f} {r[6]:8.4f} {r[7]:8.4f} {r[8]:6.3f} {r[9]:7.3f}  {mark}")

print(f"\n{n_dates := len(idx)} dates x {len(SYMBOLS)} symbols | {len(F)} candidates | "
      f"{sum(r[-1] for r in rows)} passed gate | {time.time()-t0:.1f}s")

with open("scripts/miner3_screen_cycle3_results.json", "w") as fh:
    json.dump({n: {"horizons": {str(k): v for k, v in r["horizons"].items()},
                   "coverage": r["coverage"], "turnover_10d": r["turnover_10d"]}
               for n, r in results.items()}, fh, indent=1, default=str)
print("saved scripts/miner3_screen_cycle3_results.json")
