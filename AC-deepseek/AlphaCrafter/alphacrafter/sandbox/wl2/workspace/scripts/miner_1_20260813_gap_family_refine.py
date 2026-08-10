"""miner_1 2026-08-13: gap/overnight-signal factor family refinement.

Motivation: screen showed gap_level_20 (mean overnight gap over 20d) at
IC=+0.0256, ICIR=+0.0777 (gate 0.0840) with low library correlation (max 0.31
vs gain_loss_20) and strong recent IC (last60 +0.171). Goal: find a variant of
the gap family that crosses |ICIR| >= 0.0840 while staying interpretable and
low-correlated with the library.

Construction (per-asset own calendar, reindex to union GRID -- same fix that
cured the autocorr degeneracy):
  gap_t = open_t / close_{t-1} - 1   (overnight gap)
Variants:
  gap_level_N      : mean(gap, N)                       N in {10,20,40,60}
  gap_accel_5x20   : mean(gap,5) - mean(gap,20)         gap momentum
  gap_volscale_20  : mean(gap,20) / std(daily ret,20)   vol-normalized gap
  gap_consistency_20: mean(sign(gap),20) - 0.5          direction persistence
  gap_skip1_20     : mean(gap.shift(1),20)              lagged gap level

Gates: |IC|>=0.0070 AND |ICIR|>=0.0840 at h=10; report library correlation.
No persistence in this script; it is a refinement screen.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DS = json.load(open("../persistent/date.json"))
TRADING_DAYS = DS["trading_days"]
VISIBLE = DS["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}  visible={VISIBLE}")

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = Path("../persistent/stock_data")
HORIZON = 10
MIN_ASSETS = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_own(sym, cols):
    df = pd.read_csv(DATA_DIR / f"{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return {c: pd.to_numeric(df[c], errors="coerce").astype(float) for c in cols}


OHLCV = {a: load_own(a, ["open", "close"]) for a in TRADABLES}
CLOSE_OWN = {a: OHLCV[a]["close"] for a in TRADABLES}


def reindex_panel(d):
    return pd.DataFrame(d, index=GRID)


def gap_series(df):
    return df["open"] / df["close"].shift(1) - 1.0


def build(variant):
    out = {}
    for a in TRADABLES:
        df = OHLCV[a]
        g = gap_series(df)
        r = df["close"].pct_change()
        if variant == "gap_level_10":
            s = g.rolling(10, min_periods=5).mean()
        elif variant == "gap_level_20":
            s = g.rolling(20, min_periods=10).mean()
        elif variant == "gap_level_40":
            s = g.rolling(40, min_periods=20).mean()
        elif variant == "gap_level_60":
            s = g.rolling(60, min_periods=30).mean()
        elif variant == "gap_accel_5x20":
            s = g.rolling(5, min_periods=3).mean() - g.rolling(20, min_periods=10).mean()
        elif variant == "gap_volscale_20":
            s = g.rolling(20, min_periods=10).mean() / r.rolling(20, min_periods=10).std()
        elif variant == "gap_consistency_20":
            s = np.sign(g).rolling(20, min_periods=10).mean() - 0.5
        elif variant == "gap_skip1_20":
            s = g.shift(1).rolling(20, min_periods=10).mean()
        else:
            raise ValueError(variant)
        out[a] = s.reindex(GRID)
    return pd.DataFrame(out, index=GRID)


def fwd_panel(h):
    out = {}
    for a in TRADABLES:
        s = CLOSE_OWN[a]
        out[a] = (s.shift(-h) / s - 1.0).reindex(GRID)
    return pd.DataFrame(out, index=GRID)


def ic_series(fp, rp):
    dates = fp.index.intersection(rp.index)
    Fr = fp.loc[dates].rank(axis=1).values
    Rr = rp.loc[dates].rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= MIN_ASSETS
    ics = np.full(len(dates), np.nan)
    for i in np.where(valid)[0]:
        f = Fr[i, m[i]] - Fr[i, m[i]].mean()
        r = Rr[i, m[i]] - Rr[i, m[i]].mean()
        d = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / d if d > 0 else np.nan
    return pd.Series(ics, index=dates)


def turnover_rank(fp, step=10):
    ranked = fp.rank(axis=1, pct=True)
    vals = []
    for i in range(step, len(ranked), step):
        a, b = ranked.iloc[i - step], ranked.iloc[i]
        m = a.notna() & b.notna()
        if m.sum() >= MIN_ASSETS:
            vals.append(float((b[m] - a[m]).abs().mean()))
    return float(np.mean(vals)) if vals else np.nan


# library artifacts
LIB = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    try:
        a = np.load(f, allow_pickle=True)
        if a.shape[1] == 15:
            n = min(a.shape[0], len(GRID))
            LIB[f.stem.replace(".signal", "")] = pd.DataFrame(a[:n], index=GRID[:n], columns=TRADABLES)
    except Exception:
        pass
print("library artifacts:", len(LIB))


def panel_rank_corr(a, b):
    dates = a.index.intersection(b.index)
    Ar = a.loc[dates].rank(axis=1).values
    Br = b.loc[dates].rank(axis=1).values
    m = (~np.isnan(Ar)) & (~np.isnan(Br))
    valid = m.sum(axis=1) >= MIN_ASSETS
    cs = []
    for i in np.where(valid)[0]:
        x = Ar[i, m[i]] - Ar[i, m[i]].mean()
        y = Br[i, m[i]] - Br[i, m[i]].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 0:
            cs.append((x * y).sum() / d)
    return float(np.mean(cs)) if cs else 0.0


FWD = {h: fwd_panel(h) for h in (1, 2, 3, 5, 10, 20)}
VARIANTS = ["gap_level_10", "gap_level_20", "gap_level_40", "gap_level_60",
            "gap_accel_5x20", "gap_volscale_20", "gap_consistency_20", "gap_skip1_20"]

rows = {}
for v in VARIANTS:
    F = build(v)
    ic10 = ic_series(F, FWD[10]).dropna()
    ic = float(ic10.mean())
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 1 and ic10.std() > 0 else 0.0
    hit = float((np.sign(ic10) == np.sign(ic)).mean()) if ic != 0 else 0.0
    cov_a = float(F.notna().sum().sum() / (F.shape[0] * F.shape[1]))
    cov_d = float((F.notna().sum(axis=1) >= MIN_ASSETS).mean())
    to = turnover_rank(F, HORIZON)
    libc = {fid: panel_rank_corr(F, sig) for fid, sig in LIB.items()}
    maxabs = max((abs(x) for x in libc.values()), default=0.0)
    topf = sorted(libc.items(), key=lambda kv: -abs(kv[1]))[:3]
    decay = {str(h): round(float(ic_series(F, FWD[h]).dropna().mean()), 4) for h in (1, 2, 3, 5, 10, 20)}
    passed = abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR
    rows[v] = dict(ic=ic, icir=icir, hit=hit, n=len(ic10), cov_a=cov_a, cov_d=cov_d,
                   to=to, maxlib=maxabs, top=topf, decay=decay,
                   ic250=float(ic10.iloc[-250:].mean()), ic60=float(ic10.iloc[-60:].mean()), pass_=passed)
    print(f"[{v:22s}] IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10):5d} "
          f"cov_a={cov_a:.3f} cov_d={cov_d:.3f} to={to:.3f} maxlib={maxabs:.3f} "
          f"ic250={rows[v]['ic250']:+.4f} ic60={rows[v]['ic60']:+.4f} "
          f"decay10={decay['10']:+.4f} => {'PASS' if passed else 'fail'}")
    print(f"     top lib: {', '.join(f'{k}:{x:+.3f}' for k, x in topf)}")

json.dump({v: {k: (r if k != "top" else [(a, round(b, 4)) for a, b in r]) for k, r in rows[v].items()}
           for v in rows}, open("scripts/miner_1_20260813_gap_family_results.json", "w"), indent=1)
print("\nsaved scripts/miner_1_20260813_gap_family_results.json")
