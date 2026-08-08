"""Single idea: common-stress overnight-gap resilience response, 60 observations, current cutoff."""
from pathlib import Path
src=Path('scripts/miner_1_20301128_common_stress_overnight_gap_resilience_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-11-27')", "END=pd.Timestamp('2031-01-08')")
src=src.replace("miner_1_20301128_common_stress_overnight_gap_resilience_response_60obs_candidate_signal.pkl", "miner_1_20310109_common_stress_overnight_gap_resilience_response_60obs_candidate_signal.pkl")
exec(compile(src,'gap_resilience_current_harness','exec'))
