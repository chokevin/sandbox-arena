
def strategy(prices, position, cash):
    """Parameterized trading strategy."""
    if len(prices) < 3:
        return "hold"

    window = prices[-3:]
    current = prices[-1]
    ma = sum(window) / len(window)

    # Momentum signal
    momentum = (current - prices[-3]) / prices[-3]

    # Volatility signal
    variance = sum((p - ma) ** 2 for p in window) / len(window)
    vol = variance ** 0.5
    vol_ratio = vol / ma if ma > 0 else 0

    if momentum > 0.0387 and vol_ratio < 0.0624:
        return "buy" if cash > 0 else "hold"
    elif momentum < -0.0568 or vol_ratio > 0.1884:
        return "sell" if position > 0 else "hold"
    return "hold"
