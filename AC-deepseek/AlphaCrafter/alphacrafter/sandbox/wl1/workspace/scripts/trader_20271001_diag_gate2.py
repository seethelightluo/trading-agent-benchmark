"""Compute gate-style signed edge with ACTUAL stored proposal weights."""
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

def forecasts(cur):
    ens = json.load(open("factor_ensemble.json"))["selected_factors"]
    acc = json.load(open("../persistent/account.json"))
    assets = list(acc["watch_list"])
    vixc = vix_series(cur)
    frames = {a: load_asset(a, cur) for a in assets}
    score = {a: 0.0 for a in assets}
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
    mean = float(np.mean(list(score.values())))
    half = max(1e-9, (max(score.values()) - min(score.values())) / 2.0)
    f = {}
    for a in assets:
        z = (score[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    return f, score

acc = json.load(open("../persistent/account.json"))
assets = list(acc["watch_list"])
prop = acc.get("last_proposed_target_weights")
execw = acc.get("last_executed_target_weights")
cur = "2027-10-01"
f, score = forecasts(cur)
if prop and execw:
    t2 = sum(abs(prop[a] - execw.get(a, 0.0)) for a in assets) / 2.0
    edge = sum(f[a] * (prop[a] - execw.get(a, 0.0)) for a in assets)
    edge_pos = sum(max(0.0, f[a] * (prop[a] - execw.get(a, 0.0))) for a in assets)
    cost = t2 * 0.0003
    print("== 10-01 ACTUAL stored proposal vs executed target ==")
    print(f"one-way turnover: {t2*100:.2f}%  signed_edge: {edge*100:.3f}%  pos-only edge: {edge_pos*100:.3f}%  cost: {cost*100:.4f}%")
    print(f"signed pass={edge>cost}  pos-only pass={edge_pos>cost}")
    print("per-asset delta/forecast (top by |delta|):")
    rows = sorted(assets, key=lambda a: -abs(prop[a]-execw.get(a,0.0)))
    for a in rows[:10]:
        d = prop[a] - execw.get(a, 0.0)
        print(f"  {a:10s} d={d*100:+.2f}pp f={f[a]*100:+.2f}%  score={score[a]:+.3f}")
