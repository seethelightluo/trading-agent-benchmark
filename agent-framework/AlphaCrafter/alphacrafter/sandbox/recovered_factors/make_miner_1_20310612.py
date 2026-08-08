"""One idea: inverse VIX upside-shock peer-relative resilience (40), with admitted-library novelty audit."""
import pathlib
src=pathlib.Path('scripts/miner_1_20310529_vix_upshock_peer_relative_resilience_60.py').read_text()
src=src.replace('VIX upside-shock peer-relative resilience (60)', 'Inverse VIX upside-shock peer-relative resilience (40)')
src=src.replace("cand=cs(rel.where(vix_event,axis=0).rolling(60,min_periods=12).mean()).shift(1)", "cand=cs(-rel.where(vix_event,axis=0).rolling(40,min_periods=8).mean()).shift(1)")
src=src.replace('vix_upshock_peer_relative_resilience_60', 'inverse_vix_upshock_peer_relative_resilience_40')
pathlib.Path('scripts/miner_1_20310612_inverse_vix_upshock_peer_relative_resilience_40.py').write_text(src)
print('wrote research script')
