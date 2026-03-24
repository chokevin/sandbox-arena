def strategy(prices, position, cash):
    """Contrarian — buy after 3 consecutive down days, sell after 3 up days."""
    if len(prices) < 4:
        return "hold"
    last_3 = [prices[i] - prices[i-1] for i in range(-3, 0)]
    if all(d < 0 for d in last_3):
        return "buy" if cash > 0 else "hold"
    elif all(d > 0 for d in last_3):
        return "sell" if position > 0 else "hold"
    return "hold"
