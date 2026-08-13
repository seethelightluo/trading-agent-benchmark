with open('strategy.py') as f:
    content = f.read()

old = """Ensemble (2031-10-30 refresh, quality_ic_tilt, 5f): max_consec_gain_20
.3250(+1), mom_180d_skip5 .2252(+1), downbeta_spx_60 .1707(+1), spx_corr60
.1685(+1), range_pos_252 .1106(+1). Same 5 factor IDs as the 2031-03-20 /
2031-10-02 refreshes; weights regime-updated for the elevated-plateau tape
(mom_180d x1.50 up, range_pos x1.20 up, max_consec_gain x0.85 eased,
downbeta x0.50 kept, spx_corr60 x0.80 kept). Miner gate-pass set unchanged. The 2026-11-19 10f momentum
set (mom20_volproxy60, gain_loss_20, vol_of_vol20x60,
usdjpy_beta_cond_120x60, mom30_vol60, days_since_high_60, max_consec_loss_20)
is DROPPED (miner_1 gate decay through 2030-10-16)."""

new = """Ensemble (2031-12-25 refresh, quality_ic_tilt, 5f): max_consec_gain_20
.3408(+1), mom_180d_skip5 .2305(+1), downbeta_spx_60 .1522(+1), spx_corr60
.1670(+1), range_pos_252 .1096(+1). Same 5 factor IDs as the 2031-03-20 /
2031-10-02 / 2031-10-30 refreshes; weights regime-updated for the
stabilizing-elevated tape (mom_180d x1.55 up, range_pos x1.20 kept,
max_consec_gain x0.90 eased, downbeta x0.45 kept-heavy-demoted, spx_corr60
x0.80 kept). Miner gate-pass set unchanged. The 2026-11-19 10f momentum
set (mom20_volproxy60, gain_loss_20, vol_of_vol20x60,
usdjpy_beta_cond_120x60, mom30_vol60, days_since_high_60, max_consec_loss_20)
is DROPPED (miner_1 gate decay through 2030-10-16)."""

assert old in content, "docstring anchor not found"
content = content.replace(old, new)
with open('strategy.py', 'w') as f:
    f.write(content)
print("docstring updated OK")
