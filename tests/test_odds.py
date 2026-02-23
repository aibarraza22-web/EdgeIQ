from edgeiq.odds import american_to_implied_prob, confidence_from_edge


def test_american_to_implied_prob_positive():
    assert round(american_to_implied_prob(150), 4) == 0.4


def test_american_to_implied_prob_negative():
    assert round(american_to_implied_prob(-150), 4) == 0.6


def test_confidence_bounds():
    assert confidence_from_edge(-1) == 50
    assert confidence_from_edge(20) == 85
