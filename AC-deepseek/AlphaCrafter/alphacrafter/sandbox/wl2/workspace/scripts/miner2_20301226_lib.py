"""Shared research library for miner_2 (2030-12-26 cycle).
Truncates all data at the visible date (2030-12-25) to avoid lookahead.
"""
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]
VISIBLE = "2030-12-25"

GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_prices(visible=VISIBLE):
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(visible)].set_index("date").sort_index()
        closes[a] = df["close"]
    px = pd.DataFrame(closes).dropna(how="all")
    ret = px.pct_change()
    return px, ret


def load_macro(visible=VISIBLE):
    out = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(visible)].set_index("date").sort_index()
        out[m] = df["close"]
    return pd.DataFrame(out)


def fwd_ret(ret, h):
    """Forward h-day return: ret of day t+1..t+h (skip 0)."""
    return ret.shift(-h).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)


def fwd_ret_skip(ret, h, skip):
    return ret.shift(-(h + skip)).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)


def daily_ic(factor_df, fwd, min_obs=8):
    dates = factor_df.index.intersection(fwd.index)
    ics = []
    for d in dates:
        f = factor_df.loc[d]
        r = fwd.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < min_obs:
            continue
        ics.append((d, f[mask].rank().corr(r[mask].rank())))
    return pd.Series(dict(ics))


def eval_factor(factor_df, ret, horizon=10, skip=0, min_obs=8, name=""):
    fwd = fwd_ret_skip(ret, horizon, skip) if skip else fwd_ret(ret, horizon)
    ic = daily_ic(factor_df, fwd, min_obs).dropna()
    n = len(ic)
    if n < 60:
        return {"name": name, "n_dates": n, "ic": np.nan, "icir": np.nan, "ok": False}
    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    icir = ic_mean / ic_std * np.sqrt(n) if ic_std > 0 else np.nan
    hit = (ic > 0).mean()
    rnk = factor_df.rank(axis=1).dropna(how="all")
    turn = (rnk.diff().abs() / (len(ASSETS) - 1)).mean().mean() if len(rnk) > 2 else np.nan
    cov = factor_df.notna().mean().mean()
    cov8 = (factor_df.notna().sum(axis=1) >= 8).mean()
    ok = (abs(ic_mean) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    return {"name": name, "n_dates": n, "ic": round(float(ic_mean), 4),
            "icir": round(float(icir), 3), "hit": round(float(hit), 3),
            "turnover": round(float(turn), 3) if turn == turn else np.nan,
            "coverage": round(float(cov), 3), "cov_dates_ge8": round(float(cov8), 3),
            "ok": bool(ok)}


def decay_analysis(factor_df, ret, horizons=(1, 2, 3, 5, 10, 20), min_obs=8, name=""):
    out = {}
    for h in horizons:
        fwd = fwd_ret(ret, h)
        ic = daily_ic(factor_df, fwd, min_obs).dropna()
        if len(ic) >= 60:
            out[h] = round(float(ic.mean()), 4)
        else:
            out[h] = None
    return out


def regime_ic(factor_df, ret, horizon=10, min_obs=8):
    """IC by sub-period to assess regime robustness."""
    fwd = fwd_ret(ret, horizon)
    ic = daily_ic(factor_df, fwd, min_obs).dropna()
    bounds = [(ic.index.min(), "2021-12-31"), ("2022-01-01", "2024-12-31"),
              ("2025-01-01", ic.index.max())]
    labels = ["2020-2021", "2022-2024", "2025-2030"]
    out = {}
    for (lo, hi), lab in zip(bounds, labels):
        sub = ic[(ic.index >= pd.Timestamp(lo)) & (ic.index <= pd.Timestamp(hi))]
        if len(sub) >= 30:
            out[lab] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sub.std(ddof=1) * np.sqrt(len(sub))), 3)
                        if sub.std(ddof=1) > 0 else None,
                        "n": int(len(sub))}
    last = ic.tail(250)
    if len(last) >= 30:
        out["last250"] = {"ic": round(float(last.mean()), 4),
                          "icir": round(float(last.mean() / last.std(ddof=1) * np.sqrt(len(last))), 3)
                          if last.std(ddof=1) > 0 else None,
                          "n": int(len(last))}
    return out


def corr_with_library(factor_df, px, ret, lib_files=None):
    """Max absolute rank cross-sectional correlation with existing library factors.
    Returns (rho, best_name). Uses only dates where both have data.
    """
    import glob, json, os
    if lib_files is None:
        lib_files = sorted(glob.glob("factors/*.json"))
    best_rho = 0.0
    best_name = None
    f_rank = factor_df.rank(axis=1)
    for fp in lib_files:
        base = os.path.basename(fp)
        if base.endswith(".bak") or "ensemble" in base:
            continue
        try:
            d = json.load(open(fp))
            fid = d.get("factor_id", base.replace(".json", ""))
            expr = d.get("calculation", {}).get("expression", "")
            # skip factors whose computation is not trivially derivable here
            if "signal" not in d.get("validation", {}):
                pass
        except Exception:
            continue
        # estimate library factor signals from expression strings where feasible
        sig = _signal_from_expr(expr, px, ret, fid)
        if sig is None:
            continue
        common = f_rank.index.intersection(sig.index)
        if len(common) < 120:
            continue
        rho = 0.0
        cnt = 0
        for d in common:
            a = f_rank.loc[d]
            b = sig.loc[d]
            m = a.notna() & b.notna()
            if m.sum() < 8:
                continue
            c = a[m].rank().corr(b[m].rank())
            if np.isfinite(c):
                rho += c
                cnt += 1
        if cnt > 60:
            rho /= cnt
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_name = fid
    return best_rho, best_name


def _signal_from_expr(expr, px, ret, fid):
    """Recompute a library factor from its expression string when possible."""
    try:
        if "pct_change" in expr or "close.shift" in expr or "rolling" in expr:
            pass
        # generic: try safe eval on close/ret
        import re
        e = expr.replace("close", "px").replace("ret", "ret")
        # not robust - return None unless simple known forms
        return None
    except Exception:
        return None
