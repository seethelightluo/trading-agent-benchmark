"""Reproduce strategy score/forecast/edge for past decision dates.

Reads raw CSVs directly (no account/date mutation). Replicates v7 hook
computation as-of a given decision date to see why the gate rejected the
09-17 and 10-01 proposals.
"""
import json
import math
import os
import numpy as np
import pandas as pd

DATA_DIR = "../persistent/stock_data"
VIX_FILE = "../persistent/index_data/VIX.csv"
DATA_DAYS = 170


def load_asset(symbol, cur):
    p = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(cur)].sort_values("date").tail(DATA_DAYS)
    if len(df) < 140:
        return None
    return df.set_index("date")


def vix_series(cur):
    v = pd.read_csv(VIX_FILE)
    v["date"] = pd.to_datetime(v["date"])
    v = v[v["date"] <= pd.Timestamp(cur)].sort_values("date")
    return v.set_index("date")["close"].astype(float)


def factor_series(df, fid, cur, vixc):
    o = df["open"].astype(float); h = df["high"].astype(float)
    l = df["low"].astype(float); c = df["close"].astype(float)
    if fid.endswith("nclv_1d"):
        return -(c - l) / (h - l)
    if fid.endswith("nclv_2d"):
        return -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
    if fid.endswith("nclv_3d"):
        return -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
    if fid.endswith("rev_1d"):
        return -np.log(c / c.shift(1))
    if fid.endswith("rev_2d"):
        return -np.log(c / c.shift(2))
    if fid.endswith("nbody_1d"):
        return -(c - o) / (h - l)
    if "mom_120d_skip5" in fid:
        return c.shift(5) / c.shift(125) - 1.0
    if fid == "vol_of_vol20x60":
        return c.pct_change().rolling(20).std().rolling(60).std()
    if fid == "vix_beta_cond_60x20":
        if vixc is None or len(vixc) < 90:
            return None
        v = vixc.reindex(df.index).ffill()
        ar = c.pct_change(); vr = v.pct_change()
        beta = ar.rolling(60).cov(vr) / vr.rolling(60).var()
        vm = v / v.shift(20) - 1.0
        return -beta * vm
    return None


def compute(cur, exec_weights):
    ens = json.load(open("factor_ensemble.json"))["selected_factors"]
    assets = list(exec_weights.keys())
    vixc = vix_series(cur)
    frames = {a: load_asset(a, cur) for a in assets}
    score = {a: 0.0 for a in assets}
    used = 0
    for fac in ens:
        fid, w, direction = fac["factor_id"], float(fac["weight"]), int(fac["direction"])
        vals = {}
        for a in assets:
            df = frames.get(a)
            if df is None:
                vals[a] = None; continue
            try:
                s = factor_series(df, fid, cur, vixc)
                if s is None:
                    vals[a] = None; continue
                s = s.replace([np.inf, -np.inf], np.nan)
                v = float(s.iloc[-1])
                vals[a] = v if math.isfinite(v) else None
            except Exception:
                vals[a] = None
        valid = sorted((float(v), a) for a, v in vals.items() if v is not None and math.isfinite(float(v)))
        if len(valid) < 8:
            continue
        r = {a: 0.5 for a in assets}
        n = len(valid)
        for i, (_, a) in enumerate(valid):
            r[a] = i / max(1, n - 1)
        for a in assets:
            score[a] += w * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
    mean = float(np.mean(list(score.values())))
    half = max(1e-9, (max(score.values()) - min(score.values())) / 2.0)
    f = {}
    for a in assets:
        z = (score[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    turn = sum(abs(f[a] * 0 + (0)) for a in assets)  # placeholder
    t2 = sum(abs((0.02 + 0.10 * (sorted(assets, key=lambda x: -score[x]).index(a) / 14)) - exec_weights[a]) for a in assets) / 2.0
    # rank-linear weights as v7 (pre regime/guard) - approximate
    order = sorted(assets, key=lambda a: (score[a], a))
    raw = {}
    for i, a in enumerate(order):
        raw[a] = 0.02 + 0.10 * (i / max(1, len(assets) - 1))
    tot = sum(raw.values())
    w = {a: x / tot for a, x in raw.items()}
    t2 = sum(abs(w[a] - exec_weights.get(a, 0.0)) for a in assets) / 2.0
    edge = sum(f[a] * (w[a] - exec_weights.get(a, 0.0)) for a in assets)
    cost = t2 * 0.0003
    print(f"== {cur} | factors_used={used}")
    print(f"   score std={np.std(list(score.values())):.4f} range=[{min(score.values()):+.3f},{max(score.values()):+.3f}]")
    print(f"   forecast range=[{min(f.values()):+.4f},{max(f.values()):+.4f}] n|f|>2%={sum(1 for v in f.values() if abs(v) > 0.02)}")
    print(f"   one-way turnover(approx)={t2*100:.2f}% signed_edge={edge*100:.3f}% cost={cost*100:.4f}% pass={edge > cost}")
    top = sorted(assets, key=lambda a: -score[a])[:5]
    print("   top5:", [(a, round(score[a], 3)) for a in top])


acc = json.load(open("../persistent/account.json"))
exec_w = acc.get("last_executed_target_weights") or acc.get("last_target_weights")
for cur in ["2027-09-17", "2027-10-01", "2027-10-15"]:
    compute(cur, exec_w)
