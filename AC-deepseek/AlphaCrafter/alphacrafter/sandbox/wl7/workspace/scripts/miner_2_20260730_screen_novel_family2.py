"""miner_2 cycle: screen novel factor families (skew/kurt, vol-regime, macd, stoich,
volume flow, macro-cond beta, beta asymmetry, garman-klass, drawdown). Vectorized.
Universe: 15 tradable cross-asset instruments. Gate: |IC|>=0.007 & |ICIR|>=0.084 @ h=10.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_2_lib import (WATCH, MAX_VISIBLE, FACTOR_LAST, MIN_ASSETS,
                         ADMISSION, load_panel, load_macro, per_asset,
                         fwd_returns, rank_ic_series, turnover_10d_rank)


def load_ohlcv(a: str) -> pd.DataFrame:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    return df


def full_library_signals(panel: pd.DataFrame, macro: dict) -> dict[str, pd.DataFrame]:
    rets = panel.pct_change()
    out = {}
    out["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    out["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    out["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    out["max_ret_20d"] = rets.rolling(20).max()
    out["downside_vol_ratio_20"] = -(rets.clip(upper=0).rolling(20).std() / rets.rolling(20).std())
    mkt = rets.mean(axis=1)
    out["beta_ew_60d"] = rets.rolling(60).cov(mkt) / mkt.rolling(60).var()
    vix = macro["VIX"]; vixr = vix.pct_change()
    out["vix_beta_cond_60x20"] = -(rets.rolling(60).cov(vixr) / vixr.rolling(60).var()) * (vix / vix.shift(20) - 1.0)
    amih = {}
    for a in panel.columns:
        s = panel[a]; v = load_ohlcv(a)["volume"].astype(float).reindex(panel.index)
        am = (s.pct_change().abs() / v).rolling(20).mean()
        amih[a] = am / am.rolling(252).median()
    out["amihud_20"] = pd.DataFrame(amih, index=panel.index)
    return out


def library_corr_all(factor: pd.DataFrame, panel: pd.DataFrame, macro: dict) -> tuple:
    libs = full_library_signals(panel, macro)
    per = {}
    common = factor.index.intersection(panel.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-700:]:
            f = factor.loc[dt]; g = lf.loc[dt]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def evaluate(name: str, factor: pd.DataFrame, panel: pd.DataFrame, macro: dict) -> dict:
    factor_w = factor.loc[:FACTOR_LAST]
    fwd = fwd_returns(panel, 10)
    ic = rank_ic_series(factor_w, fwd)
    direction = float(np.sign(ic.mean())) if np.isfinite(ic.mean()) and ic.mean() != 0 else 1.0
    ic = ic * direction
    icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    hit = float((ic > 0).mean()) if len(ic) else float("nan")
    valid = factor_w.notna()
    cov_ad = float(valid.mean().mean())
    cov_d8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    to = turnover_10d_rank(factor_w)
    max_corr, per = library_corr_all(factor_w, panel, macro)
    res = dict(name=name, n_dates=int(len(ic)), ic_h10=float(ic.mean()), icir_h10=icir,
               hit_h10=hit, cov_ad=cov_ad, cov_d8=cov_d8, turnover=to,
               max_corr=max_corr, per=per, direction=direction,
               pass_=bool(abs(float(ic.mean())) >= ADMISSION["ic"] and abs(icir) >= ADMISSION["icir"]))
    print(f"[{name}] ic10={res['ic_h10']:+.4f} icir10={res['icir_h10']:+.4f} "
          f"hit={hit:.3f} n={res['n_dates']} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} "
          f"to={to:.3f} max_corr={max_corr:.3f} PASS={res['pass_']}")
    return res


def rolling_beta_cond(panel: pd.DataFrame, x: pd.Series, mask: pd.Series, W: int) -> pd.DataFrame:
    """Vectorized rolling beta of each asset vs x, restricted to days where mask is True."""
    out = {}
    rets = panel.pct_change()
    m = mask.reindex(panel.index).fillna(False).astype(float)
    for a in panel.columns:
        r = rets[a].fillna(0.0)
        n = m.rolling(W).sum()
        sx = (r * m).rolling(W).sum()
        sy = (x * m).rolling(W).sum()
        sxy = (r * x * m).rolling(W).sum()
        syy = (x * x * m).rolling(W).sum()
        denom = n * syy - sy * sy
        num = n * sxy - sx * sy
        out[a] = (num / denom.replace(0, np.nan)).where(n >= 30)
    return pd.DataFrame(out, index=panel.index)


if __name__ == "__main__":
    panel = load_panel()
    rets = panel.pct_change()
    macro = load_macro()
    results = {}

    results["skew_60"] = evaluate("skew_60", per_asset(lambda s: s.pct_change().rolling(60).skew())(panel, macro), panel, macro)
    results["skew_20"] = evaluate("skew_20", per_asset(lambda s: s.pct_change().rolling(20).skew())(panel, macro), panel, macro)
    results["kurt_60"] = evaluate("kurt_60", per_asset(lambda s: s.pct_change().rolling(60).kurt())(panel, macro), panel, macro)
    results["vol_mom_20x60"] = evaluate("vol_mom_20x60", per_asset(
        lambda s: s.pct_change().rolling(20).std() / s.pct_change().rolling(60).std() - 1.0)(panel, macro), panel, macro)

    def macd(s):
        e12 = s.ewm(span=12, adjust=False).mean(); e26 = s.ewm(span=26, adjust=False).mean()
        return (e12 - e26) / s
    results["macd_12x26"] = evaluate("macd_12x26", per_asset(macd)(panel, macro), panel, macro)

    def stoich(s):
        lo = s.rolling(20).min(); hi = s.rolling(20).max()
        return (s - lo) / (hi - lo)
    results["stoich_20"] = evaluate("stoich_20", per_asset(stoich)(panel, macro), panel, macro)

    vf = {}
    for a in panel.columns:
        v = load_ohlcv(a)["volume"].astype(float).reindex(panel.index)
        vf[a] = v.rolling(5).mean() / v.rolling(60).mean() - 1.0
    results["vol_flow_5x60"] = evaluate("vol_flow_5x60", pd.DataFrame(vf, index=panel.index), panel, macro)

    dxy = macro["DXY"]; dxy_r = dxy.pct_change()
    dxy_beta = rets.rolling(60).cov(dxy_r) / dxy_r.rolling(60).var()
    results["dxy_beta_cond_60x20"] = evaluate("dxy_beta_cond_60x20", -dxy_beta * (dxy / dxy.shift(20) - 1.0), panel, macro)

    cny = macro["USDCNY"]; cny_r = cny.pct_change()
    cny_beta = rets.rolling(60).cov(cny_r) / cny_r.rolling(60).var()
    results["cny_beta_cond_60x20"] = evaluate("cny_beta_cond_60x20", -cny_beta * (cny / cny.shift(20) - 1.0), panel, macro)

    # up/down beta asymmetry vs EW market
    mkt = rets.mean(axis=1)
    bu = rolling_beta_cond(panel, mkt, mkt > 0, 60)
    bd = rolling_beta_cond(panel, mkt, mkt <= 0, 60)
    results["beta_asym_60"] = evaluate("beta_asym_60", bu - bd, panel, macro)

    gk = {}
    for a in panel.columns:
        df = load_ohlcv(a)
        o, h, l, c = df["open"].astype(float), df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        gk[a] = (0.5 * (np.log(h / l) ** 2) - (2 * np.log(2) - 1) * (np.log(c / o) ** 2)).rolling(20).mean().apply(np.sqrt)
    results["gk_vol_20"] = evaluate("gk_vol_20", pd.DataFrame(gk, index=panel.index), panel, macro)

    results["dd_60"] = evaluate("dd_60", per_asset(lambda s: s / s.rolling(60).max() - 1.0)(panel, macro), panel, macro)
    results["rev_10d_skip1"] = evaluate("rev_10d_skip1", -(panel.shift(1) / panel.shift(11) - 1.0), panel, macro)
    results["mom_5d_skip1"] = evaluate("mom_5d_skip1", panel.shift(1) / panel.shift(6) - 1.0, panel, macro)

    json.dump(results, open("scripts/miner_2_cycle5_results.json", "w"), indent=1, default=float)
    print("\nSaved cycle5 results.")
