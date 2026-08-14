"""miner_1 exploration 2034-11-17: scan candidate factor families.
Reads data through visible_through only (no lookahead). Prints IC/ICIR for each
candidate at horizons 1/5/10, full-sample and last-2y window."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')
from miner1_20341117_loader import (load_calendar, load_panel, load_macro,
                                    build_close_matrix, forward_returns, rank_ic)

cur, vis, tdays = load_calendar()
panel = load_panel(vis)
macro = load_macro(vis)
cm = build_close_matrix(panel)
cl = np.log(cm)

# macro series aligned to cm index
vix = macro['VIX'].set_index('date')['close']
vix = vix.reindex(cm.index).ffill()
dxy = macro['DXY'].set_index('date')['close']
dxy = dxy.reindex(cm.index).ffill()
us10y = cm['US10Y']
cn10y = cm['CN10Y']

rets = cm.pct_change()

def zscore_ts(s):
    return (s - s.mean()) / (s.std() + 1e-12)

# ---------- candidate factories ----------
def make_pair_rot(nd=5):
    """Intra-pair relative momentum: asset 5d ret minus pair-partner 5d ret.
    Pairs: BTC-ETH, WTI-COPPER, 000300-000688, SPX-NDX, SOX-NDX, US10Y-CN10Y, N225-SX5E, HSI-000300."""
    pairs = [("BTC","ETH"),("WTI","COPPER"),("000300.SH","000688.SH"),
             ("SPX","NDX"),("SOX","NDX"),("US10Y","CN10Y"),("N225","SX5E"),("HSI","000300.SH")]
    # map each symbol to its pair partner (first occurrence)
    partner = {}
    for a,b in pairs:
        partner.setdefault(a,b); partner.setdefault(b,a)
    out = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    r = cm.pct_change(nd)
    for s in cm.columns:
        p = partner.get(s)
        if p is None:
            out[s] = np.nan
        else:
            out[s] = r[s] - r[p]
    return out

def make_crisis_mom(nd=5, vix_thr=60.0, skip=1):
    """Short-horizon momentum active only in crisis (VIX>thr), else 0."""
    r = cm.pct_change(nd).shift(skip)
    crisis = (vix > vix_thr).astype(float)
    out = r.mul(crisis, axis=0)
    return out

def make_crisis_rev_guard(nd=5, vix_thr=60.0):
    """Reversal that is damped for crypto in crisis: crypto value=0 when VIX>thr."""
    base = -(np.log(cm).diff(nd))
    crisis = (vix > vix_thr).astype(float)
    out = base.copy()
    for s in ["BTC","ETH"]:
        out[s] = base[s] * (1 - crisis)
    return out

def make_dxy_beta_cond(beta_win=60, dxy_win=20):
    """Asset beta to DXY * 20d DXY move (like vix_beta but DXY)."""
    dxy_ret = dxy.pct_change()
    out = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    for s in cm.columns:
        a = rets[s]
        b = dxy_ret
        beta = a.rolling(beta_win).cov(b) / b.rolling(beta_win).var()
        dxy_move = dxy.pct_change(dxy_win)
        out[s] = beta * dxy_move
    return out

def make_rate_beta_cond(beta_win=60, rate_win=20):
    """Asset beta to US10Y price * US10Y 20d price move."""
    r10 = us10y.pct_change()
    out = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    for s in cm.columns:
        a = rets[s]
        beta = a.rolling(beta_win).cov(r10) / r10.rolling(beta_win).var()
        rmove = us10y.pct_change(rate_win)
        out[s] = beta * rmove
    return out

def make_spillover(nd=5):
    """Lead-lag spillover: follower = leader's nd-ret. Followers: SOX<-NDX, ETH<-BTC,
    COPPER<-WTI, 000688<-000300, NDX<-SPX. Leaders keep own nd-ret."""
    rel = {"SOX":"NDX","ETH":"BTC","COPPER":"WTI","000688.SH":"000300.SH","NDX":"SPX"}
    r = cm.pct_change(nd)
    out = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    for s in cm.columns:
        lead = rel.get(s)
        out[s] = r[lead] if lead else r[s]
    return out

def make_range_z(short=20, long=60):
    """z-score of 20d mean daily range vs 60d history (range expansion)."""
    rng = (cm['high'] - cm['low']) / cm['close'] if 'high' in cm.columns else None
    # build range from panel
    ranges = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    for s in panel:
        df = panel[s].set_index('date')
        df = df.reindex(cm.index).ffill()
        ranges[s] = (df['high'] - df['low']) / df['close']
    m20 = ranges.rolling(short).mean()
    m60 = ranges.rolling(long).mean()
    s60 = ranges.rolling(long).std()
    out = (m20 - m60) / (s60 + 1e-9)
    return out

def make_vol_scaled_mom(nd=5, vol_win=20):
    """nd-momentum scaled by inverse vol (Sharpe-like short momentum)."""
    r = cm.pct_change(nd)
    vol = cm.pct_change().rolling(vol_win).std()
    return r / (vol + 1e-9)

# ---------- evaluate ----------
def evaluate(name, fac):
    fac = fac.reindex(cm.index)
    fwd1 = forward_returns(cm, 1)
    fwd5 = forward_returns(cm, 5)
    fwd10 = forward_returns(cm, 10)
    ic1 = rank_ic(fac, fwd1)
    ic5 = rank_ic(fac, fwd5)
    ic10 = rank_ic(fac, fwd10)
    full = ic1.index >= pd.Timestamp('2021-01-01')
    rec = ic1.index >= pd.Timestamp('2032-11-16')
    def summ(s):
        if len(s) < 30: return f"n={len(s)}"
        m, sd = s.mean(), s.std(ddof=1)
        return f"n={len(s)} ic={m:+.4f} icir={m/sd:+.3f} hit={(s>0).mean():.2f}"
    cov = fac.notna().mean().mean()
    print(f"[{name}] coverage={cov:.3f}")
    print(f"   ic1 full {summ(ic1[full])} | recent2y {summ(ic1[rec])}")
    print(f"   ic5 full {summ(ic5[full])} | recent2y {summ(ic5[rec])}")
    print(f"   ic10 full {summ(ic10[full])} | recent2y {summ(ic10[rec])}")
    return ic1

cands = {}
for nd in [3,5]:
    cands[f"pair_rot_{nd}d"] = make_pair_rot(nd)
for nd in [3,5,10]:
    cands[f"crisis_mom_{nd}d_vix60"] = make_crisis_mom(nd)
cands["crisis_rev_guard_5d_vix60"] = make_crisis_rev_guard(5)
cands["dxy_beta_cond_60x20"] = make_dxy_beta_cond()
cands["rate_beta_cond_60x20"] = make_rate_beta_cond()
for nd in [3,5]:
    cands[f"spillover_{nd}d"] = make_spillover(nd)
cands["range_z_20x60"] = make_range_z()
for nd in [3,5]:
    cands[f"volscaled_mom_{nd}d"] = make_vol_scaled_mom(nd)

print("="*60)
for k, v in cands.items():
    evaluate(k, v)
