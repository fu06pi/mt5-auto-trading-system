import types

from test_xauusd_trend_strategy_risk_controls import load_strategy_module, make_config


def make_strategy(tmp_path, **overrides):
    module = load_strategy_module()
    config = make_config(module, tmp_path, **overrides)
    return module.XAUUSDTrendStrategy(config)


def test_live_quality_filter_blocks_compensated_h1_when_raw_h1_disagrees(tmp_path):
    strategy = make_strategy(tmp_path, require_raw_htf_agree=True)

    allowed = strategy._quality_filter_allows(
        signal="BUY",
        htf_signal="BEAR",
        compensated_htf_signal="BULL",
        score=1.10,
        adx=30.0,
    )

    assert not allowed


def test_live_quality_filter_requires_minimum_absolute_score(tmp_path):
    strategy = make_strategy(tmp_path, min_abs_score=0.95)

    allowed = strategy._quality_filter_allows(
        signal="BUY",
        htf_signal="BULL",
        compensated_htf_signal="BULL",
        score=0.94,
        adx=30.0,
    )

    assert not allowed


def test_live_quality_filter_requires_minimum_adx(tmp_path):
    strategy = make_strategy(tmp_path, min_adx=22.0)

    allowed = strategy._quality_filter_allows(
        signal="SELL",
        htf_signal="BEAR",
        compensated_htf_signal="BEAR",
        score=-1.10,
        adx=21.9,
    )

    assert not allowed


def test_live_quality_filter_allows_high_quality_raw_aligned_signal(tmp_path):
    strategy = make_strategy(
        tmp_path,
        min_abs_score=0.95,
        min_adx=22.0,
        require_raw_htf_agree=True,
    )

    allowed = strategy._quality_filter_allows(
        signal="SELL",
        htf_signal="BEAR",
        compensated_htf_signal="BEAR",
        score=-1.20,
        adx=30.0,
    )

    assert allowed


def test_parser_exposes_live_quality_filter_flags():
    module = load_strategy_module()
    args = module.build_parser().parse_args(
        [
            "--min-abs-score",
            "0.95",
            "--min-adx",
            "22",
            "--require-raw-htf-agree",
        ]
    )

    assert args.min_abs_score == 0.95
    assert args.min_adx == 22.0
    assert args.require_raw_htf_agree is True
