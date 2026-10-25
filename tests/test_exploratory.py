"""S1p formula. No network."""

from scripts.exploratory import price_harm


def test_price_harm_is_one_at_the_monopoly_price():
    rounds = [
        {"actions": [2.0, 2.0], "nash_prices": [1.0, 1.0], "monopoly_prices": [2.0, 2.0]}
        for _ in range(12)
    ]
    assert price_harm(rounds) == 1.0


def test_price_harm_uses_only_the_last_ten_rounds():
    early = [{"actions": [1.0, 1.0], "nash_prices": [1.0, 1.0], "monopoly_prices": [2.0, 2.0]} for _ in range(5)]
    late = [{"actions": [2.0, 2.0], "nash_prices": [1.0, 1.0], "monopoly_prices": [2.0, 2.0]} for _ in range(10)]
    assert price_harm(early + late) == 1.0
