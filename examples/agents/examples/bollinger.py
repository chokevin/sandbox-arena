def strategy(prices, position, cash):
    """Bollinger Band strategy — buy below lower band, sell above upper band."""
    if len(prices) < 20:
        return "hold"
    window = prices[-20:]
    mean = sum(window) / len(window)
    variance = sum((p - mean) ** 2 for p in window) / len(window)
    std = variance ** 0.5
    upper = mean + 2 * std
    lower = mean - 2 * std
    current = prices[-1]
    if current < lower:
        return "buy" if cash > 0 else "hold"
    elif current > upper:
        return "sell" if position > 0 else "hold"
    return "hold"
