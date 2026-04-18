import pandas as pd
import numpy as np
from student_t_bvc import compute_sigma_causal_session_isolated

def test_session_isolated_sigma():
    # Create synthetic data
    # 2 contracts, with a time gap in the first contract
    n_bars = 500
    
    # Contract 1: 0 to 199
    ts1 = pd.date_range('2026-01-01 09:30:00', periods=200, freq='5min', tz='UTC')
    # Gap of 2 hours, then 200 to 399
    ts2 = pd.date_range(ts1[-1] + pd.Timedelta(hours=2), periods=200, freq='5min', tz='UTC')
    # Contract 2: 400 to 499, overlaps in time with ts2 but different instrument_id
    ts3 = pd.date_range(ts2[-1] - pd.Timedelta(hours=1), periods=100, freq='5min', tz='UTC')
    
    ts_event = ts1.append(ts2).append(ts3)
    instrument_id = [1]*400 + [2]*100
    
    df = pd.DataFrame({
        'ts_event': ts_event,
        'instrument_id': instrument_id,
        'open': np.ones(500),
        'high': np.ones(500) * 1.01,
        'low': np.ones(500) * 0.99,
        'close': np.ones(500),
        'volume': np.ones(500) * 100
    })
    
    warmup = 40
    span = 20
    gap = '60min'
    
    # Run function (new tuple return)
    sigma, warmup_valid = compute_sigma_causal_session_isolated(df, span=span, warmup_bars=warmup, gap_threshold=gap)
    df['sigma'] = sigma.values
    df['warmup_valid'] = warmup_valid.values

    # Test 1: warmup bars are NaN
    assert df['sigma'].iloc[0:warmup].isna().all(), "Warmup bars not NaN in session 1"
    assert not df['sigma'].iloc[warmup:200].isna().any(), "Valid bars are NaN in session 1"

    assert df['sigma'].iloc[200:200+warmup].isna().all(), "Warmup bars not NaN after gap"
    assert not df['sigma'].iloc[200+warmup:400].isna().any(), "Valid bars are NaN after gap"

    assert df['sigma'].iloc[400:400+warmup].isna().all(), "Warmup bars not NaN after instrument change"
    assert not df['sigma'].iloc[400+warmup:500].isna().any(), "Valid bars are NaN after instrument change"

    # Test 1b: warmup_valid flag matches exactly the transition at bar warmup_bars
    for session_start in (0, 200, 400):
        pre = df['warmup_valid'].iloc[session_start:session_start + warmup]
        post = df['warmup_valid'].iloc[session_start + warmup:session_start + warmup + 10]
        assert (~pre).all(), f"warmup_valid should be False in warmup window at {session_start}"
        assert post.all(), f"warmup_valid should be True after bar {warmup} at {session_start}"

    # Test 2: State doesn't carry across boundary
    df_fresh = df.iloc[200:400].copy()
    sigma_fresh, valid_fresh = compute_sigma_causal_session_isolated(df_fresh, span=span, warmup_bars=warmup, gap_threshold=gap)

    np.testing.assert_array_almost_equal(
        df['sigma'].iloc[200:400].values,
        sigma_fresh.values,
        err_msg="State carried across time gap boundary"
    )
    np.testing.assert_array_equal(
        df['warmup_valid'].iloc[200:400].values,
        valid_fresh.values,
        err_msg="warmup_valid inconsistent between in-context and fresh session"
    )

    print("All unit tests passed for compute_sigma_causal_session_isolated.")

if __name__ == '__main__':
    test_session_isolated_sigma()
