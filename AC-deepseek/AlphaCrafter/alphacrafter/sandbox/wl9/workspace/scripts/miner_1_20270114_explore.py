# miner_1 2027-01-14 factor exploration (novel cross-asset factors)
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from miner3_20260730_harness import (
    load_closes, forward_returns, rank_ic,
    weekday_grid, ASSETS, STOCK_DIR,
)

FACTOR_DIR = Path("factors")
VALID_START, FACTOR_END = "2020-01-01", "2027-01-13"

closes = load_closes()  # uses VISIBLE_END 2026-07-29 internally; re-extend below

# extended data through 2027-01-13
#define local loader to extend window
closes_ext = {}
for a in ASSETS:
    f = STOCK_DIR / f"{a}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= pd.Timestamp("2027-01-13")]
    closes_ext[a] = df.set_index("date")["close"].astype(float)


def grid_ext(closes_):
    dates = sorted({d for s in closes_.values() for d in s.index})
    dates = [d for d in dates if VALID_START <= d.strftime("%Y-%m-%d") <= FACTOR_END and d.weekday() < 5]
    return pd.DatetimeIndex(dates)


GRID = grid_ext(closes_ext)
print("window factor dates:", GRID.min().date(), "..", GRID.max().date())


def to_frame_ext(values):
    df = pd.DataFrame(index=GRID, columns=ASSETS, dtype=float)
    for a, s in values.items():
        if a in df.columns:
            df[a] = s.reindex(GRID)
    return df


def full_eval(label, values, h=10):
    frame = to_frame_ext(values)
    ret_ext = {a: closes_ext[a].shift(-h) / closes_ext[a] - 1.0 for a in closes_ext}
    ret_frame = pd.DataFrame({a: ret_ext[a].reindex(GRID) for a in ret_ext})
    ic = rank_ic(frame, ret_frame, h)
    n = len(ic)
    icm = float(ic.mean()) if n else float("nan")
    ics = float(ic.std(ddof=1)) if n > 2 else float("nan")
    icir = (icm / ics) if ics and np.isfinite(ics) else float("nan")
    hit = float((np.sign(ic) == np.sign(icm)).mean()) if n and icm != 0 else float("nan")
    cover_dates = float((frame.notna().sum(axis=1) >= 8).mean())
    passed = abs(icm) >= 0.0070 and abs(icir) >= 0.0840
    print(f"{label:<24} IC={icm:>8.4f} ICIR={icir:>8.4f} hit={hit:>5.3f} n={n:>5d} "
          f"cov8={cover_dates:>4.2f} PASS={passed}")
    return dict(label=label, ic=icm, icir=icir, hit=hit, n=n, passed=passed,
                frame=frame, ic_series=ic)


# equal-weight cross-asset composite return
comp_ret = pd.DataFrame({a: closes_ext[a].pct_change() for a in closes_ext}).mean(axis=1, skipna=True)


def rolling_beta(ar, cr, N):
    return ar.rolling(N).cov(cr) / cr.rolling(N).var()


def per_asset(fn, N):
    out = {}
    for a in closes_ext:
        r = closes_ext[a].pct_change()
        out[a] = fn(r, N, a)
    return out


def mk_beta(N):
    def f(r, N, a):
        return rolling_beta(r, comp_ret.reindex(r.index), N)
    return per_asset(f, N)


def mk_vol_of_mom(N):
    def f(r, N, a):
        return r.rolling(N).std() * (abs(r.rolling(N).sum()) + 1e-12)
    return per_asset(f, N)


def mk_idio_vol(N):
    def f(r, N, a):
        beta = rolling_beta(r, comp_ret.reindex(r.index), N)
        resid = r - beta * comp_ret.reindex(r.index)
        return resid.rolling(N).std()
    return per_asset(f, N)


def mk_mom_diverg(N):
    def f(r, N, a):
        mom = r.rolling(N).sum()
        return mom - comp_ret.reindex(r.index).rolling(N).sum()
    return per_asset(f, N)


def mk_rng_proxy(N):
    def f(r, N, a):
        c = closes_ext[a]
        rng = c.rolling(N).max() - c.rolling(N).min()
        return rng / rng.rolling(N).mean()
    return per_asset(f, N)


for N in (60, 90):
    full_eval(f"beta_{N}d", mk_beta(N))
for N in (60,):
    full_eval(f"vol_of_mom_{N}d", mk_vol_of_mom(N))
for N in (20, 60):
    full_eval(f"idio_vol_{N}d", mk_idio_vol(N))
for N in (60,):
    full_eval(f"mom_diverg_{N}d", mk_mom_diverg(N))
for N in (20,):
    full_eval(f"range_proxy_{N}d", mk_rng_proxy(N))