"""Trader patch 2029-06-08: sync ensemble weights (vol_of_vol .23->.21,
nclv_1d .15->.17) in docstring and add v13 composite-rank top-2 cap."""
import re

P = "strategy.py"
src = open(P).read()

# 1) Docstring header + ensemble description sync
old_hdr = '''"""Trader strategy v12 - Screener 5-factor quality_ic_tilt ensemble.

Ensemble (2028-11-10): mom_120d_skip5 (.26,+) | vol_of_vol20x60 (.23,+)
| vix_beta_cond_60x20 (.20,-) | miner2_20260715_rev_2d (.16,+)
| miner2_20260715_nclv_1d (.15,+). Screener re-tilted momentum up (.22->.26)
on a regime shift toward persistent momentum leaders with extreme dispersion
while trimming the vol/defensive block (vol_of_vol .25->.23, vix_beta .24
->.20); reversal pair reweighted (rev_2d .13->.16, nclv_1d .16->.15).'''
new_hdr = '''"""Trader strategy v13 - Screener 5-factor quality_ic_tilt ensemble.

Ensemble (2029-06-08 sync): mom_120d_skip5 (.26,+) | vol_of_vol20x60 (.21,+)
| vix_beta_cond_60x20 (.20,-) | miner2_20260715_nclv_1d (.17,+)
| miner2_20260715_rev_2d (.16,+). Screener 2029-06-08 re-tilted the
vol/reversal block (vol_of_vol .23->.21, nclv_1d .15->.17) to fund the
best-quality 1d reversal factor (IC1 .065, ICIR1 .183) in the high-vol
whipsaw regime; momentum anchor and vix_beta unchanged. Minimal sync -
continuity preserved (Screener: "do not structurally change it").'''
assert old_hdr in src, "header block not found"
src = src.replace(old_hdr, new_hdr)

# 2) Append v13 note after the v12 docstring paragraph
v12_end = '''whipsaw block in 10 (Screener also flags XAU/BTC as top-4 trap names in 20d
downtrends). Any of the top-2 momentum names (rank >= .86 of 15) is now
capped at GUARD_CAP regardless of MA state, matching the empirical 9/10
top-pick reversal rate.
"""'''
v13_note = '''whipsaw block in 10 (Screener also flags XAU/BTC as top-4 trap names in 20d
downtrends). Any of the top-2 momentum names (rank >= .86 of 15) is now
capped at GUARD_CAP regardless of MA state, matching the empirical 9/10
top-pick reversal rate.

v13 (2029-06-08): composite-rank top-2 weight cap (Screener-recommended).
Block 0330-0413: NDX 11.5% / N225 10.8% were top-2 COMPOSITE weights but not
top-2 momentum names, so the v12 momentum cap missed them; 2029-04-13
feedback + 2029-06-08 screener both recommend a composite-rank cap beyond the
momentum-rank guard. With 10 of the last 11 blocks showing a top-pick
whipsaw (SOX -12.5%, WTI -18.4% twice, ETH -25.9%...), composite-score
leaders carry fat-tailed reversal risk regardless of factor origin. The
top-2 composite-score names are capped at COMP_TOP2_CAP (9.5%); excess is
redistributed proportionally to remaining names. Factor-agnostic like
v8/v9/v11; applied before the MA guards.
"""'''
assert v12_end in src, "v12 docstring end not found"
src = src.replace(v12_end, v13_note)

# 3) Add COMP_TOP2_CAP constant after COMMOD_CAP
old_const = "COMMOD_CAP = 0.14        # v11 combined WTI+COPPER weight cap"
new_const = old_const + "\nCOMP_TOP2_CAP = 0.095     # v13 composite-rank top-2 weight cap"
assert old_const in src, "COMMOD_CAP const not found"
src = src.replace(old_const, new_const)

# 4) Insert _composite_top2_cap function before _de_rank_value_traps
anchor = "def _de_rank_value_traps(scores, frames, assets, cur):"
fn = '''def _composite_top2_cap(w, assets, scores):
    """v13: cap the top-2 composite-score names at COMP_TOP2_CAP (9.5%).

    Block 0330-0413: NDX 11.5% / N225 10.8% were top-2 COMPOSITE weights but
    not top-2 momentum names, so the v12 momentum cap missed them. With 10 of
    the last 11 blocks showing a top-pick whipsaw, composite-score leaders
    carry fat-tailed reversal risk regardless of factor origin (momentum OR
    reversal lifted). Factor-agnostic like v8/v9/v11; excess is redistributed
    proportionally to the remaining names.
    """
    order = sorted(assets, key=lambda a: (scores[a], a))
    top2 = set(order[-2:])
    for _ in range(80):                    # iterate until cap invariant holds
        penalized = {a for a in top2 if w[a] > COMP_TOP2_CAP + 1e-9}
        if not penalized:
            break
        excess = sum(w[a] - COMP_TOP2_CAP for a in penalized)
        for a in penalized:
            w[a] = COMP_TOP2_CAP
        room = [a for a in assets if a not in penalized]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


'''
assert anchor in src, "de_rank anchor not found"
src = src.replace(anchor, fn + anchor)

# 5) Insert call in strategy_hook after _weights
old_call = "    w = _weights(scores, assets, regime)\n    w = _composite_ma_guard(w, frames, assets)                  # v8 (8% cap)"
new_call = "    w = _weights(scores, assets, regime)\n    w = _composite_top2_cap(w, assets, scores)                 # v13 (9.5% top-2)\n    w = _composite_ma_guard(w, frames, assets)                  # v8 (8% cap)"
assert old_call in src, "hook call anchor not found"
src = src.replace(old_call, new_call)

open(P, "w").write(src)
print("patched OK")
print("COMP_TOP2_CAP lines:", sum(1 for l in src.splitlines() if "COMP_TOP2_CAP" in l))
