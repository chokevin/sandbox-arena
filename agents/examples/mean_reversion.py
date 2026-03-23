def strategy(prices, position, cash):
    """Mean reversion — buy when price dips below moving average, sell when above."""
    if len(prices) < 10:
        return "hold"
    ma = sum(prices[-10:]) / 10
    if prices[-1] < ma * 0.98:
        return "buy" if cash > 0 else "hold"
    elif prices[-1] > ma * 1.02:
        return "sell" if position > 0 else "hold"
    return "hold"
