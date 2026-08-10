"""miner_2 2026-07-30: Explore volatility-asymmetry / drawdown / autocorrelation factor family.

Candidates (h=10, 15-asset cross-asset universe, min_valid=8):
  A. downside_vol_ratio_60 : downside semideviation / total vol (asymmetry)
  B. drawdown_depth_60     : 60d max drawdown depth (close/rolling_max - 1)
  C. skew_ret_60           : skewness of 60d daily returns
  D. max_gain_loss_60      : max daily gain / |min daily loss| over 60d
  E. autocorr_ret_5        : 5-day lag autocorrelation of daily returns
  F. updown_vol_ratio_60   : vol on up-days vs down-days (directional asymmetry)

Also computes max-abs IC-series correlation vs all recoverable quarantined
library artifacts (mom, beta, vol factors persisted earlier this worldline).
"""
import sys, json, glob, base64, zlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10

close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)
print(f"close panel: {close.shape}, dates {close.index.min().date()}..{close.index.max().date()}")


def build_candidates():
    cands = {}

    # A. downside semideviation ratio
    neg = ret.clip(upper=0)
    pos = ret.clip(lower=0)
    down_sd = (neg ** 2).rolling(60).mean().apply(np.sqrt)
    tot_sd = ret.rolling(60).std()
    cands["downside_vol_ratio_60"] = down_sd / tot_sd

    # B. drawdown depth over 60d
    roll_max = close.rolling(60).max()
    cands["drawdown_depth_60"] = close / roll_max - 1.0

    # C. skewness of 60d returns
    cands["skew_ret_60"] = ret.rolling(60).skew()

    # D. max gain vs |max loss| over 60d
    max_g = ret.rolling(60).max()
    max_l = ret.rolling(60).min()
    cands["max_gain_loss_60"] = max_g / max_l.abs()

    # E. 5-day lag autocorrelation of daily returns
    def autocorr_lag(s, lag=5, win=60):
        out = {}
        for a in s.columns:
            x = s[a]
            r = x.rolling(win).corr(x.shift(lag))
            out[a] = r
        return pd.DataFrame(out).reindex(s.index)
    cands["autocorr_ret_5"] = autocorr_lag(ret, 5, 60)

    # F. up-day vol vs down-day vol (directional asymmetry)
    up_vol = ret.where(ret > 0).rolling(60).std()
    dn_vol = ret.where(ret < 0).rolling(60).std()
    cands["updown_vol_ratio_60"] = up_vol / dn_vol

    return cands


def load_quarantine_ic():
    """Recover IC series of every quarantined factor that carries a signal artifact."""
    lib = {}
    for fp in sorted(glob.glob("factors/quarantine/*.json")):
        if fp.endswith("reason.json"):
            continue
        try:
            d = json.load(open(fp))
            art = d.get("validation", {}).get("signal_artifact")
            if not art:
                continue
            dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
            sig = pd.read_csv(io.StringIO(dec), index_col=0)
            sig.index = pd.to_datetime(sig.index)
            s = ic_series(sig.reindex(close.index), fr, min_valid=8)
            if len(s) > 30:
                lib[d["factor_id"]] = s
        except Exception as e:
            print("  skip artifact:", fp, e)
    return lib


lib_ic = load_quarantine_ic()
print(f"recoverable library IC series: {list(lib_ic.keys())}")

cands = build_candidates()
results = {}
for name, f in cands.items():
    f = f.reindex(close.index)
    ic = ic_series(f, fr, min_valid=8)
    m = summary_metrics(ic, f, fr, close, h=H)
    if m is None:
        print(f"\n=== {name}: insufficient IC dates ===")
        continue
    m["regime"] = regime_split(ic)
    rho = {}
    for lid, lic in lib_ic.items():
        pair = pd.concat([ic.rename("a"), lic.rename("b")], axis=1).dropna()
        rho[lid] = round(float(pair["a"].corr(pair["b"])), 4) if len(pair) > 30 else None
    m["rho_vs_library"] = rho
    m["rho_max"] = max([abs(v) for v in rho.values() if v is not None] or [0.0])
    results[name] = {"ic": ic, "factor": f, "metrics": m}
    print(f"\n=== {name} ===")
    print(f" IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']}")
    print(f" cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']}")
    print(f" decay={m['decay_ic_by_horizon']}")
    print(f" regime={json.dumps(m['regime'])}")
    print(f" rho_max={m['rho_max']} rho={json.dumps(rho)}")

print("\n--- gate check (IC>=0.007, ICIR>=0.084) ---")
for name, r in results.items():
    mm = r["metrics"]
    g1 = abs(mm["ic"]) >= 0.007
    g2 = abs(mm["icir"] or 0) >= 0.084
    print(f"{name}: IC={mm['ic']:.4f} ICIR={mm['icir']} rho_max={mm['rho_max']:.3f} PASS_IC={g1} PASS_ICIR={g2}")
