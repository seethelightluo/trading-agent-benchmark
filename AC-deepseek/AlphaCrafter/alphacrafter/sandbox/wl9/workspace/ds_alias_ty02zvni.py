import subprocess
# view recent miner_1 scripts' docstrings to know what was tried
for fn in ['scripts/miner_1_20281019_probe_data.py','scripts/miner_1_20280615_probe_panel.py','scripts/miner_1_20280420_fx_beta_differential.py','scripts/miner_1_20280309_probe_data.py','scripts/miner_1_20280210_sweep.py']:
    lines = open(fn).read().splitlines()
    doc = ' '.join(l.strip() for l in lines[:3] if l.strip() and not l.strip().startswith(('import','from','"""','#')))
    print(f"{fn}:\n  {doc[:220]}")