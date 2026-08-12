import sys; sys.path.insert(0,'scripts')
import inspect
import factor_research_lib as frl
for fn in ['close_panel','ret_panel','forward_returns','rank_ic_series','summarize_ic','coverage_metrics','turnover_rank','decay_profile','max_library_corr','library_signals','load_library_meta']:
    try:
        src = inspect.getsource(getattr(frl, fn))
        print(f"====={fn}=====")
        print(src[:1200])
    except Exception as e:
        print(fn, "ERR", e)