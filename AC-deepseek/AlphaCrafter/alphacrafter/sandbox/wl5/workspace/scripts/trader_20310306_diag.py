"""Diagnose strategy proposal: replicate factor pipeline as of current sim date."""
import json, sys
sys.path.insert(0, '.')
from math import isfinite, copysign
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

# ---- replicate strategy internals ----
def _load_ensemble():
    try:
        p = Path("factors/factor_ensemble.json")
        import pathlib
        p = pathlib.Path("factors/factor_ensemble.json")
        ens = json.loads(p.read_text())
        out = []
        for it in ens.get("selected_factors", []):
            if not isinstance(it, dict):
                continue
            fid = str(it.get("factor_id", ""))
            if not fid:
                continue
            try:
                w = float(it.get("weight", 0.0))
                d = int(it.get("direction", 1))
            except (TypeError, ValueError):
                continue
            out.append((fid, w, d))
        if out:
            return out[:10]
    except Exception as e:
        print("load err", e)
    return None

FETCH = 200
def _closes(assets):
    out = {}
    for a in assets:
        df = None
        try:
            df = get_stock_daily_data(a, days=FETCH)
        except Exception:
            df = None
        if df is None or len(df) < 140:
            try:
                df = get_index_daily_data(a, days=FETCH)
            except Exception:
                df = None
        if df is not None and len(df) >= 140 and "close" in df:
            s = df[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"])
            out[a] = s.set_index("date")["close"].astype(float)
    return out

def _macro_close(symbol):
    df = None
    try:
        df = get_index_daily_data(symbol, days=150)
    except Exception:
        df = None
    if df is None or "close" not in df or len(df) < 80:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)

def _rank_map(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out

acct = get_account_dict()
assets = list(acct["watch_list"])
print("sim current date check via data tail:")
closes = _closes(assets)
print("n closes:", len(closes))
panel = pd.DataFrame(closes).sort_index()
print("panel date range:", panel.index[0].date(), "->", panel.index[-1].date())
rets = panel.pct_change()

dxy_c = _macro_close("DXY"); dxy_r = dxy_c.pct_change() if dxy_c is not None else None
vix_c = _macro_close("VIX"); vix_r = vix_c.pct_change() if vix_c is not None else None
cny_c = _macro_close("USDCNY"); cny_r = cny_c.pct_change() if cny_c is not None else None
print("macro available:", dxy_c is not None, vix_c is not None, cny_c is not None)

# import actual strategy functions
import importlib.util
spec = importlib.util.spec_from_file_location("strat", "strategy.py")
# importing strategy.py would trigger register_hook import - instead copy funcs via module exec w/o hook side effects
import pathlib
src = pathlib.Path("strategy.py").read_text()
# strip the decorator call
src2 = src.replace("@register_hook", "#@register_hook")
ns = {"__name__": "strat_diag"}
exec(compile(src2, "strategy_diag.py", "exec"), ns)

ensemble = ns["_load_ensemble"]()
print("\nloaded ensemble (from json):")
for fid, w, d in ensemble:
    print(f"  {fid} w={w:.4f} dir={d:+d}")
print("sum w:", round(sum(w for _, w, _ in ensemble), 4))

fvals = {fid: {} for fid, _, _ in ensemble}
for a in assets:
    c = closes.get(a)
    r = rets[a] if a in rets else None
    if c is None or r is None:
        continue
    for fid, _, _ in ensemble:
        try:
            if fid == "trend_r2_30_signed":
                v = ns["_trend_r2"](c)
            elif fid == "semi_down_ratio_20":
                v = ns["_semi_down_ratio"](r)
            elif fid == "mom_120d_skip5":
                v = ns["_mom_120"](c)
            elif fid == "mom_10d_skip5":
                v = ns["_mom_10"](c)
            elif fid == "vol_of_vol20x60":
                v = ns["_vol_of_vol"](r)
            elif fid == "time_under_water_120":
                v = ns["_underwater"](c)
            elif fid == "tail_ratio_20":
                v = ns["_tail_ratio"](r)
            elif fid == "dxy_beta_60":
                v = ns["_beta_60"](r, dxy_r) if dxy_r is not None else None
            elif fid == "cny_beta_60":
                v = ns["_beta_60"](r, cny_r) if cny_r is not None else None
            elif fid == "vix_beta_cond_60x20":
                v = ns["_vix_beta_cond"](r, vix_r, vix_c) if vix_r is not None else None
            else:
                v = None
        except Exception as e:
            v = None
        fvals[fid][a] = v

print("\nper-factor non-null counts:")
for fid, _, _ in ensemble:
    n = sum(1 for v in fvals[fid].values() if v is not None)
    print(f"  {fid}: {n}/15")

score = {a: 0.0 for a in assets}
for fid, w, direction in ensemble:
    rk = _rank_map(fvals[fid], assets)
    for a in assets:
        score[a] += w * direction * rk[a]

sv = np.array([score[a] for a in assets])
print("\nscore min/max/std:", round(float(sv.min()),4), round(float(sv.max()),4), round(float(sv.std()),5))
for a in sorted(assets, key=lambda x: -score[x]):
    print(f"  {a}: {score[a]:+.4f}")

market = rets.mean(axis=1)
trend20 = float(market.tail(20).mean())
avg_px = float(panel.mean(axis=1).iloc[-1])
ma60 = float(panel.mean(axis=1).tail(60).mean())
print("\nregime: trend20=", round(trend20,5), "avg_px=", round(avg_px,3), "ma60=", round(ma60,3), "bearish=", trend20 < 0 and avg_px < ma60)

weights = ns["_to_weights"](score, assets, {a: (2.2 if (trend20<0 and avg_px<ma60 and a in {"XAU","US10Y","CN10Y"}) else (0.75 if trend20<0 and avg_px<ma60 else 1.0)) for a in assets})
print("\nproposed weights (v21):")
for a in sorted(assets, key=lambda x: -weights[x]):
    print(f"  {a}: {weights[a]:.4f}")
print("sum:", round(sum(weights.values()),6))
