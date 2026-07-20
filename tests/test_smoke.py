from cricketiq.core import config

def test_temporal_split_sane():
    assert config.TRAIN_END_YEAR < config.VAL_YEAR < config.TEST_START_YEAR

def test_phases_cover_20_overs():
    covered = sorted(o for r in config.PHASES.values() for o in r)
    assert covered == list(range(1, 21))