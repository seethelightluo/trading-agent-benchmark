"""miner_2 2026-12-21: focused validation of the drawdown-reversal factor family.
One idea: rolling max-drawdown depth predicts forward returns (mean-reversion in
the cross-asset universe). Sweep lookback windows {20,40,60,90,120} to pick the
most robust variant. Window 2020-01-01..2026-12-18, gates h=10:
|IC|>=0.0070, |ICIR|>=0.0840, max abs spearman library corr < 0.5.
Sign: factor value = min(close/rolling_max - 1) over window (<=0); deeper
drawdown (lower value) => higher fwd ret => expected_direction = -1."""
from __future__ import annotations
import sys, json, base64, zlib, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, TRADABLE,
                                 coverage_metrics, turnover_rank)

HORIZON = 10
MIN_VALID = 8
END = pd.Timestamp("2026-12-18")
WINDOW = (pd.Timestamp("2020-01-01"), END)
HORIZONS = (1, 2, 3, 5, 10, 20)
SUB_PERIODS = {"full": (pd.Timestamp("2020-01-01"), END),
               "2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
               "2021": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
               "2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
               "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
               "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
               "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
               "2026": (pd.Timestamp("2026-01-01"), END),
               "online": (pd.Timestamp("2026-07-16"), END)}


def rank_ic_vec(F: pd.DataFrame, R: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    common = F.index.intersection(R.index)
    Fr = F.loc[common].rank(axis=1, method="average")
    Rr = R.loc[common].rank(axis=1, method="average")
    X = Fr.values.astype(float)
    Y = Rr.values.astype(float)
    valid = ~(np.isnan(X) | np.isnan(Y))
    n = valid.sum(axis=1)
    keep = n >= min_valid
    X, Y, V, N = X[keep], Y[keep], valid[keep], n[keep]
    Xv = np.where(V, X, np.nan)
    Yv = np.where(V, Y, np.nan)
    Xc = X - np.nanmean(Xv, axis=1, keepdims=True)
    Yc = Y - np.nanmean(Yv, axis=1, keepdims=True)
    Xc = np.where(V, Xc, 0.0)
    Yc = np.where(V, Yc, 0.0)
    xy = (Xc * Yc).sum(axis=1)
    xx = (Xc * Xc).sum(axis=1)
    yy = (Yc * Yc).sum(axis=1)
    denom = np.sqrt(xx * yy)
    ok = (xx > 1e-14) & (yy > 1e-14) & (denom > 0)
    ic = np.full(len(X), np.nan)
    ic[ok] = xy[ok] / denom[ok]
    return pd.Series(ic, index=Fr.index[keep], name="ic").dropna()


def summarize_ic(ic_series: pd.Series, expected_sign: int = -1):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4),
            "ic_hit_ratio": round(hit, 3), "n_ic_dates": int(len(ic_series)),
            "ic_std": round(sd, 4)}


t0 = time.time()
panels = load_panels(3000)
closes_all = close_panel(panels)
clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}
idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
fwd = closes.shift(-HORIZON) / closes - 1.0

# load library factors
lib = {}
for p in sorted(Path("factors").glob("*.json")):
    if p.name == "factor_ensemble.json":
        continue
    try:
        d = json.loads(p.read_text())
        sa = d.get("validation", {}).get("signal_artifact")
        if not sa:
            continue
        fmt = sa.get("format")
        if fmt == "base64:zlib:csv":
            raw = zlib.decompress(base64.b64decode(sa["data"]))
            df = pd.read_csv(io.BytesIO(raw), index_col=0)
            df.index = pd.to_datetime(df.index)
        elif fmt == "panel_json_v1":
            df = pd.DataFrame(sa["values"], index=pd.to_datetime(sa["dates"]), columns=sa["assets"])
        else:
            continue
        lib[d["factor_id"]] = df.reindex(closes.index)
    except Exception as e:
        print(f"skip lib {p.name}: {e}")


def spearman_lib_corr(cand_df, lib):
    best, best_key = 0.0, None
    cstack = cand_df.stack().rename("c")
    for name, lib_df in lib.items():
        both = pd.concat([cstack, lib_df.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["c"].corr(both["l"], method="spearman"))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


def max_dd_panel(win: int):
    out = {}
    for a, s in clean.items():
        roll_max = s.rolling(win).max()
        dd = s / roll_max - 1.0
        out[a] = dd.rolling(win).min()
    return pd.DataFrame(out).reindex(closes_all.index).loc[idx]


results = {}
for win in (20, 40, 60, 90, 120):
    panel = max_dd_panel(win)
    ics = rank_ic_vec(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, -1)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {str(h): round(float(rank_ic_vec(panel, closes.shift(-h) / closes - 1.0, MIN_VALID).mean()), 4)
                                for h in HORIZONS}
    sub = {}
    for sname, (s0, s1) in SUB_PERIODS.items():
        sub_ics = rank_ic_vec(panel.loc[s0:s1], fwd.loc[s0:s1], MIN_VALID)
        sub[sname] = round(float(sub_ics.mean()), 4) if len(sub_ics) else None
    corr, key = spearman_lib_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    results[win] = m
    print(f"\n=== max_drawdown_{win}d === IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} covge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']} decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} }")
    print(f"    subperiods: { {k: v for k,v in sub.items()} }")
    print(f"    spearman lib corr={corr:.4f} ({key})  |  gates: IC={abs(m['ic'])>=0.007} ICIR={abs(m['icir'])>=0.084} CORR={corr<0.5}")

print("\n===== WINDOW SWEEP SUMMARY (expected direction: deeper drawdown => higher fwd ret, sign -1) =====")
for win, m in results.items():
    ok = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and m["max_abs_library_correlation"] < 0.5
    print(f"{'PASS' if ok else 'FAIL'} win={win:3d} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} corr={m['max_abs_library_correlation']:.4f}")
print(f"\ntotal time {time.time()-t0:.1f}s")
