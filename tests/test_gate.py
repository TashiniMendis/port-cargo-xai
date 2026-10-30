from src.gate import decide


def test_first_model_passes():
    assert decide(7000.0, None, 0.0)


def test_better_or_equal_passes():
    assert decide(6900.0, 7000.0, 0.0)
    assert decide(7000.0, 7000.0, 0.0)


def test_worse_fails_unless_within_tolerance():
    assert not decide(7100.0, 7000.0, 0.0)
    assert decide(7100.0, 7000.0, 0.02)
